import numpy as np

from copy import copy

class DataLoader:
    def __init__(self, dataset, shuffle=False, batch_size=1):
        self.dataset = dataset
        self.length = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        class DataIter:
            def __init__(self, data_loader):
                self.data_loader = data_loader
                self.iteration = 0
                self.indices = np.arange(0, self.data_loader.length, dtype=int)
                if self.data_loader.shuffle:
                    np.random.shuffle(self.indices)

            def __iter__(self):
                return self

            def __next__(self):
                dl = self.data_loader
                if self.iteration >= dl.length:
                    raise StopIteration
                elif dl.batch_size == 1:
                    data = dl.dataset[self.indices[self.iteration]]
                    self.iteration += 1
                    return data
                else:
                    # Get start and end of batch (end might be past end of indices)
                    begin = self.iteration
                    end = self.iteration + dl.batch_size
                    
                    # Get indices and thus slice of data
                    inds = self.indices[begin:end]

                    # Add number of indices to iteration count
                    # (will take into account size of dataset)
                    self.iteration += len(inds)
                    
                    # Return list of data
                    return [dl.dataset[i] for i in inds]

        return DataIter(self)

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))