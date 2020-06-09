import data
from cnn_model import DementiaAnalysisModel


if __name__ == '__main__':
    #Getting database and labels
    print("Getting database and labels")
    database, labels = data.get_databases()
    
    model = DementiaAnalysisModel()
    model.train(database, labels, 30)