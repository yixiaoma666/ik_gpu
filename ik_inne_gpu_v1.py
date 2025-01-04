import torch
import warnings
import numpy as np
import tqdm


class IK_inne_gpu():
    def __init__(self,
                 t,
                 psi,
                 ) -> None:
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available.")
        else:
            self.device = torch.device("cuda")
        self._t = t
        self._psi = psi
        self._center_index_list = None
        self._radius_list = None
        self.X: np.ndarray = None
        
    def fit(self,
            X: np.ndarray,
            ) -> None:
        self.X = X
        
        if self._psi > X.shape[0]:
            self._psi = X.shape[0]
            warnings.warn(f"psi is set to {X.shape[0]} as it is greater than the number of data points.")
        
        # shape=(t, psi)
        self._center_index_list = np.array([np.random.permutation(X.shape[0])[:self._psi] for _ in range(self._t)])
        self._center_list = np.zeros((self._psi * self._t, X.shape[1]))
        
        self._radius_list = torch.zeros((self._t, self._psi), dtype=torch.float32, device=self.device)

        
        for i in range(self._t):
            sample = self.X[self._center_index_list[i]]
            self._center_list[i*self._psi:(i+1)*self._psi] = sample
            s2s = torch.cdist(torch.tensor(sample, dtype=torch.float32, device=self.device), torch.tensor(sample, dtype=torch.float32, device=self.device), p=2)
            s2s.fill_diagonal_(float('inf'))
            self._radius_list[i] = torch.min(s2s, dim=0).values

        return

    def transform(self,
                  X: np.ndarray,
                  batch_size: int = 10000,
                  ) -> np.ndarray:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        output = torch.zeros((X.shape[0], self._psi*self._t), device="cpu", dtype=torch.int8)
        
        batch_num = np.ceil(X.shape[0] / batch_size).astype(int)
        
        for _batch_num in tqdm.trange(batch_num):
            start_idx = _batch_num * batch_size
            end_idx = (_batch_num + 1) * batch_size if _batch_num < batch_num - 1 else X.shape[0]
            
            batch_cuda = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=self.device)
            
            
            for i in range(self._t):
                sample = self._center_list[i*self._psi:(i+1)*self._psi]
                sample_cuda = torch.tensor(sample, dtype=torch.float32, device=self.device)

                p2s = torch.cdist(batch_cuda, sample_cuda, p=2)

                p2ns_index = torch.argmin(p2s, dim=1)
                p2ns = p2s[torch.arange(batch_cuda.shape[0], device=self.device), p2ns_index]
                ind = p2ns < self._radius_list[i, p2ns_index]
                output[start_idx:end_idx, :][ind, (p2ns_index + i * self._psi)[ind]] = 1

        if X.shape[0] == 1:
            return output.reshape(-1)

        return output.numpy()

