import itertools
import torch.nn
import torch.nn.functional
import math
from geoopt import PoincareBall
import geoopt


def mobius_linear(
    input,
    weight,
    manifold,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None
    # c=1.0,
):
    if hyperbolic_input:
        output = manifold.mobius_matvec(weight, input)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = manifold.expmap0(output)
    if bias is not None:
        if not hyperbolic_bias:
            bias = manifold.expmap0(bias)
        output = manifold.mobius_add(output, bias)
    if nonlin is not None:
        output = manifold.mobius_fn_apply(nonlin, output)
    # output = manifold.project(output)
    return output


def one_rnn_transform(W, h, U, x, b, manifold):
    W_otimes_h = manifold.mobius_matvec(W, h)
    U_otimes_x = manifold.mobius_matvec(U, x)
    Wh_plus_Ux = manifold.mobius_add(W_otimes_h, U_otimes_x)
    return manifold.mobius_add(Wh_plus_Ux, b)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    manifold,
    # c: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = manifold.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, manifold)).sigmoid()
    r_t = manifold.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, manifold)).sigmoid()

    rh_t = manifold.mobius_pointwise_mul(r_t, hx)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, manifold)

    if nonlin is not None:
        h_tilde = manifold.mobius_fn_apply(nonlin, h_tilde)
    delta_h = manifold.mobius_add(-hx, h_tilde)
    h_out = manifold.mobius_add(hx, manifold.mobius_pointwise_mul(z_t, delta_h))
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    manifold,
    # c: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
):
    if not hyperbolic_hidden_state0:
        hx = manifold.expmap0(h0)
    else:
        hx = h0
    if not hyperbolic_input:
        input = manifold.expmap0(input)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                manifold=manifold
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                manifold=manifold,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


class MobiusLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        c=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=c)
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(self.ball.expmap0(self.bias.normal_() / 4))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            manifold=self.ball,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()

class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=c)
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = self.ball.expmap0(point)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        input = input.unsqueeze(-2)
        # TODO: find right version of dis2plane
        distance = self.ball.dist2plane(
            x=input, p=self.point, a=self.tangent, c=self.ball.c, signed=True
        )
        return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}, "
            "c={ball.c}".format(
                **self.__dict__
            )
        )


class MobiusGRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlin=None,
        hyperbolic_input=True,
        hyperbolic_hidden_state0=True,
        c=1.0,
    ):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.weight_ih = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_hh = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = geoopt.ManifoldParameter(
                    self.ball.expmap0(bias), manifold=self.ball
                )
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input
        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                manifold=self.ball,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)
