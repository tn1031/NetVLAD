import chainer
from chainer import functions as F
from chainer import initializers as I
from chainer import links as L
from chainer import Parameter

class AttentionVLADpooling(chainer.Chain):
    def __init__(self, D, K):
        super(AttentionVLADpooling, self).__init__()
        self.D = D
        self.K = K
        initializer = I._get_initializer(None)

        with self.init_scope():
            self.wb = L.Convolution2D(D, K, ksize=1, stride=1, pad=0)
            self.c  = Parameter(initializer, shape=(D, K))
            self.attn = L.Convolution2D(D, 1, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        bs, channel, width, height = x.shape
        assert channel == self.D
        N = width * height

        # assignment
        a = self.wb(x)
        a = F.softmax(a)
        a = F.reshape(a, (bs, self.K, N))
        a = F.stack([a] * self.D, axis=1)

        # attention
        w = F.relu(self.attn(x))
        w = F.reshape(w, (bs, 1, N))
        w = F.stack([w] * self.D * self.K)
        w = F.reshape(w, (bs, self.D, self.K, N))

        x = F.reshape(x, (bs, self.D, N))
        x = F.stack([x] * self.K, axis=2)

        _c = F.broadcast_to(
            F.stack([self.c] * N, axis=2), (bs, self.D, self.K, N))

        v = F.sum(w * a * (x - _c), axis=3)

        v = F.normalize(v, axis=2)
        v = F.reshape(v, (bs, self.D * self.K))
        v = F.normalize(v, axis=1)
        return v
