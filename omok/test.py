import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import time

class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        load_model_path = 'model/model.ckpt'

        # 바둑판 크기
        board_size = 15
        self.game_end = 0
        self.board_size = board_size
        self.board = np.zeros([board_size,board_size])
        self.board_history = np.zeros([board_size,board_size])
        self.cnt = 1


        time_now = time.gmtime(time.time())
        self.save_name = str(time_now.tm_year) + '_' + str(time_now.tm_mon) + '_' + str(time_now.tm_mday) + '_' + str(time_now.tm_hour) + '_' + str(time_now.tm_min) + '_' + str(time_now.tm_sec) + '.txt'
        self.save_name_png = str(time_now.tm_year) + '_' + str(time_now.tm_mon) + '_' + str(time_now.tm_mday) + '_' + str(time_now.tm_hour) + '_' + str(time_now.tm_min) + '_' + str(time_now.tm_sec) + '.png'

        
        # read image in numpy array (using cv2)
        board_cv2 = cv2.imread('source/board_1515.png')
        self.board_cv2 = cv2.cvtColor(board_cv2, cv2.COLOR_BGR2RGB)

        white_ball = cv2.imread('source/white.png')
        self.white_ball = cv2.cvtColor(white_ball, cv2.COLOR_BGR2RGB)

        black_ball = cv2.imread('source/black.png')
        self.black_ball = cv2.cvtColor(black_ball, cv2.COLOR_BGR2RGB)

        # numpy to QImage
        height, width, channel = self.board_cv2.shape
        bytesPerLine = 3 * width
        qImg_board = QImage(self.board_cv2.data, width, height, bytesPerLine, QImage.Format_RGB888)

        self.player = 1 # 1: 흑  / 2: 백
        x = 0
        y = 0

        self.lbl_img = QLabel()
        self.lbl_img.setPixmap(QPixmap(qImg_board))

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.lbl_img)
        self.setLayout(self.vbox)

        self.setWindowTitle('오목 시뮬레이션')
        self.move(100, 100)
        self.resize(500,500)
        self.show()

    def mousePressEvent(self, e):
        x = e.x()
        y = e.y()
        
        self.board_cv2 = self.game_play(self.board_cv2, self.black_ball, y, x, 1)
        save_name =  'result/' + str(self.cnt) + "board_black.png"
        save_name_w = 'result/' + str(self.cnt) + "board_white.png"
        save_name_pred = 'result/' + str(self.cnt) + "board_pred.png"
        #cv2.imwrite(save_name, save_image)

        height, width, channel = self.board_cv2.shape
        bytesPerLine = 3 * width
        qImg_board = QImage(self.board_cv2.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.lbl_img.setPixmap(QPixmap(qImg_board))
        print((x-10)//55,(y-5)//55)

    def game_play(self, board_img, ball, pos_x, pos_y, turn):
        #human

        ball_size = ball.shape[0]
        step_size = 56
        off_set = 10

        # 판의 마지막 모서리에는 돌을 두지 못하게 한다.
        if pos_x < step_size/2+off_set+1 or pos_y < step_size/2+off_set+1:
            print('그곳에는 둘 수 없습니다')

        elif pos_x > step_size*self.board_size+step_size/2+off_set or pos_y > step_size*self.board_size+step_size/2+off_set:
            print('그곳에는 둘 수 없습니다')

        else:

            step_x = round((pos_x - off_set)/step_size)
            step_y = round((pos_y - off_set)/step_size)

            if self.board[step_x-1,step_y-1] != 0: # 이미 돌이 있을때
                print('그곳에는 둘 수 없습니다')

            else:
                self.board[step_x-1,step_y-1] = turn
                self.board_history[step_x-1,step_y-1] = self.cnt
                self.cnt = self.cnt + 1
                
                x_step = step_size*step_x-round(step_size/2) + off_set
                y_step = step_size*step_y-round(step_size/2) + off_set
                
                board_img[x_step:x_step+ball_size,y_step:y_step+ball_size] = ball
         
        return board_img

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())