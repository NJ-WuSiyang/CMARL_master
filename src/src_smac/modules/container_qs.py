import torch.nn as nn


class ContainerQs(nn.Module):
    def __init__(self, args):
        super(ContainerQs, self).__init__()
        self.args = args
        self.fc = nn.Conv1d(in_channels=args.n_container * args.rnn_hidden_dim,
                            out_channels=args.n_container * args.n_actions,
                            kernel_size=1, groups=args.n_container)

        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        r'''
        :param hidden_states: shape of (batch_size * n_agent, hidden_dim)
        :return: q values under different strategy: shape of (batch_size * n_agent, n_container, n_actions)
        '''
        reshaped_hidden_states = hidden_states.unsqueeze(-1).repeat(1, self.args.n_container, 1)
        # reshaped_hidden_states shape: (batch_size * n_agent, n_container * hidden_dim, 1)
        q = self.fc(reshaped_hidden_states)
        # q shape: (batch_size * n_agent, n_container * n_actions, 1)
        return q.view(q.shape[0], self.args.n_container, -1)

    def load(self, container_id, fc):
        self.fc.weight.data[
            container_id * self.args.n_actions: (container_id + 1) * self.args.n_actions, :, 0
        ] = fc.weight.data.clone().detach().to(self.fc.weight.data.device)
        self.fc.bias.data[
            container_id * self.args.n_actions: (container_id + 1) * self.args.n_actions
        ] = fc.bias.data.clone().detach().to(self.fc.bias.data.device)
