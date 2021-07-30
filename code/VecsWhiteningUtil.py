"""
最重要的使用场景，计算kernel和bias和数据尽可能和实际场景使用的数据保持一致
"""
import numpy as np
from sklearn.preprocessing import normalize


# TODO 如何增量
class VecsWhiteningUtil():
    @staticmethod
    def get_model_kernel_bias(model, sens):
        vecs = model.get_sens_vec(sens)
        return VecsWhiteningUtil.compute_kernel_bias(vecs)

    @staticmethod
    def compute_kernel_bias(vecs: np.ndarray, n_components: int = None):
        """
        计算kernel和bias
        最后的变换：y = (x + bias).dot(kernel)
        :param vecs:ndarray
        :param n_components:int,降维数量，不指定则不降维
        :return:kernel,bias
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(s ** 0.5))
        W = np.linalg.inv(W.T)
        if W is not None:
            W = W[:, :n_components]
        return W, -mu

    @staticmethod
    def transform_and_normalize(vecs, kernel, bias):
        """
        应用变换，然后标准化
        :param vecs: 待标准化向量
        :param kernel:kernel
        :param bias:bias
        :return:标注化后的向量
        """
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)
        return normalize(vecs)


if __name__ == "__main__":
    kernel, bias = VecsWhiteningUtil.compute_kernel_bias(np.random.random((300000, 768)))
    print(kernel.shape, bias.shape)
    new_vec = VecsWhiteningUtil.transform_and_normalize(np.random.random((32, 768)), kernel, bias)
    print(new_vec.shape)
