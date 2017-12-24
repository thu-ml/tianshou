import os
import cPickle

class Data(object):
    def __init__(self):
        self.boards = []
        self.probs = []
        self.winner = 0

def file_to_training_data(file_name):
    with open(file_name, 'rb') as file:
        try:
            file.seek(0)
            data = cPickle.load(file)
            return data.winner
        except Exception as e:
            print(e)
            return 0

if __name__ == "__main__":
    win_count = [0, 0, 0]
    file_list = os.listdir("./data")
    #print file_list
    for file in file_list:
        win_count[file_to_training_data("./data/" + file)] += 1
    print "Total play : " + str(len(file_list))
    print "Black wins : " + str(win_count[1])
    print "White wins : " + str(win_count[-1])

