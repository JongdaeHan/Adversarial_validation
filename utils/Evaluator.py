class Evaluator():
    def __init__(self, label, positive, negative, true_positive, false_negative, true_negative, false_positive):
        self.label = str(label)
        self.positive = positive
        self.negative = negative

        # pred -> postive
        self.true_positive = true_positive
        self.false_negative = false_negative

        # pred -> negative
        self.true_negative = true_negative
        self.false_positive = false_positive

    # TODO ::
    # 1. 음수 데이터에 대한 처리 -> 음수가 나오면 안돼 ?
    # 2. divided by zero에 대한 처리
    def get_accuracy(self):
        """
            각 라벨에 대한 정확도를 계산하는 함수, 정확도 = 전체 데이터 중 제대로 예측한 데이터의 비율
            입력 : positive, negative, true_positive, true_negative
            출력 : 정확도(%)
        """
        accuracy = (self.true_positive + self.true_negative) / (self.positive + self.negative) * 100
        print(self.label + '\'s accuracy : %.2f' % accuracy + '%')
        return accuracy

    def get_error_rate(self):
        """
            각 라벨에 대한 오탐율을 계산하는 함수, 오탐율 = 전체 데이터 중 잘못 예측한 데이터의 비율
            입력 : positive, negative, false_negative, false_positive
            출력 : 오탐율(%)
        """
        error_rate = (self.false_negative + self.false_positive) / (self.positive + self.negative) * 100
        print(self.label + '\'s error rate : %.2f' % error_rate + '%')
        return error_rate

    def get_sensitivity(self):
        """
            각 라벨에 대한 민감도를 계산하는 함수, 민감도 = 전체 positive 중 실제로 positive로 제대로 분류된 데이터의 비율
            입력 : positive, true_positive
            출력 : 민감도(%)
        """
        sensitivity = self.true_positive / self.positive * 100
        print(self.label + '\'s sensitivity : %.2f' % sensitivity + '%')
        return sensitivity

    def get_precision(self):
        """
            각 라벨에 대한 정확도를 계산하는 함수, 정확도 = 전체 positive로 분류된 데이터 중 실제로 positive인 데이터의 비율
            입력 : true_positive, false_positive
            출력 : 정확도(%)
        """
        precision = self.true_positive / (self.true_positive + self.false_positive) * 100
        print(self.label + '\'s precision : %.2f' % precision + '%')
        return precision

    def get_specificity(self):
        """
            각 라벨에 대한 특이성을 계산하는 함수, 특이성 = negative로 분류된 데이터 중 실제 negative인 데이터의 비율
            입력 : negative, true_negative
            출력 : 특이성
        """
        specificity = self.true_negative / self.negative * 100
        print(self.label + '\'s specificity : %.2f' % specificity + '%')
        return specificity

    def get_false_positive_rate(self):
        """
            각 라벨에 대한 거짓부정율을 계산하는 함수, 거짓부정율 = 전체 negative 중 negative인데 positive로 분류된 데이터의 비율
            입력 : negative, false_positive 
        """
        false_positive_rate = self.false_positive / self.negative * 100
        print(self.label + '\'s false positive rate : %.2f' % false_positive_rate + '%')
        return false_positive_rate

# for method testing
if __name__ == '__main__':
    evaluator = Evaluator('1', 50, 50, 30, 20, 30, 20)

    acc = evaluator.get_accuracy() # 60
    err = evaluator.get_error_rate() # 40
    sn = evaluator.get_sensitivity() # 60
    prec = evaluator.get_precision() # 60
    sp = evaluator.get_specificity() # 30
    fpr = evaluator.get_false_positive_rate() # 40