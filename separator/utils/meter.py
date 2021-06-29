class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def result(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count
