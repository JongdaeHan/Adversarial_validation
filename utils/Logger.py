import datetime

class Logger():
    def __init__(self, code_type='train', model='vgg16', dataset='CIFAR10'):
        self.dir = './logs/' + code_type + '/'
        self.date = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
        self.title = self.date + '-' + code_type + '-' + model + '-' + dataset + '.txt'

    def create_txt(self, hyper_parameter_infos, logs):
        """
            log 파일 기록 함수
            입력 :
                - hyper_parameter_infos : 실험에 사용한 하이퍼 파라미터 정보
                - logs : 학습 또는 평가 코드 실행 기록
            출력 : 
                - 실험에 사용한 정보들과 실험 기록이 저장된 txt 파일
        """
        f = open(self.dir + self.title, "w")
        f.write('# Hyper-parameter informations\n')
        f.write(hyper_parameter_infos + '\n\n')
        f.write('# Logs\n')
        f.write(logs)
        f.close()

if __name__ == '__main__':
    logger = Logger('train', 'vgg16')

    print(logger.title)
    logger.create_txt('hyper', 'logs')
