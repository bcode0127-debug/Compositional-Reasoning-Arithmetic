"""Manual per-timestep unroll of the bidirectional LSTM encoder, to recover
per-position CELL states (c_t) - nn.LSTM only exposes per-position HIDDEN
states via `outputs`; the returned `cell` is only the final-timestep state
per direction, not one per position.

Validated by reconstructing per-position HIDDEN states the same way and
checking they match nn.LSTM's own `outputs` tensor exactly.
"""
import torch


def manual_lstm_unroll(lstm_module, embedded):
    """embedded: [batch, seq, input_size]. Returns h_fwd, c_fwd, h_bwd, c_bwd,
    each [batch, seq, hidden_size], all in ORIGINAL left-to-right position order."""
    device = embedded.device
    batch, seq_len, input_size = embedded.shape
    hidden_size = lstm_module.hidden_size

    W_ih = lstm_module.weight_ih_l0
    W_hh = lstm_module.weight_hh_l0
    b_ih = lstm_module.bias_ih_l0
    b_hh = lstm_module.bias_hh_l0
    W_ih_r = lstm_module.weight_ih_l0_reverse
    W_hh_r = lstm_module.weight_hh_l0_reverse
    b_ih_r = lstm_module.bias_ih_l0_reverse
    b_hh_r = lstm_module.bias_hh_l0_reverse

    def step(x_t, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh):
        gates = x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    # forward direction: t = 0 .. seq_len-1
    h_fwd = torch.zeros(batch, seq_len, hidden_size, device=device)
    c_fwd = torch.zeros(batch, seq_len, hidden_size, device=device)
    h_prev = torch.zeros(batch, hidden_size, device=device)
    c_prev = torch.zeros(batch, hidden_size, device=device)
    for t in range(seq_len):
        h_prev, c_prev = step(embedded[:, t, :], h_prev, c_prev, W_ih, W_hh, b_ih, b_hh)
        h_fwd[:, t, :] = h_prev
        c_fwd[:, t, :] = c_prev

    # backward direction: processes t = seq_len-1 .. 0; h_bwd[t]/c_bwd[t] is the
    # state after having consumed positions [seq_len-1 .. t] in that order.
    h_bwd = torch.zeros(batch, seq_len, hidden_size, device=device)
    c_bwd = torch.zeros(batch, seq_len, hidden_size, device=device)
    h_prev = torch.zeros(batch, hidden_size, device=device)
    c_prev = torch.zeros(batch, hidden_size, device=device)
    for t in reversed(range(seq_len)):
        h_prev, c_prev = step(embedded[:, t, :], h_prev, c_prev, W_ih_r, W_hh_r, b_ih_r, b_hh_r)
        h_bwd[:, t, :] = h_prev
        c_bwd[:, t, :] = c_prev

    return h_fwd, c_fwd, h_bwd, c_bwd
