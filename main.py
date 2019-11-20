import measure
import ProcessImage as image
from train import train
import cv2
from train_k_neighbours import train as train_neigh
import matplotlib.pyplot as plt
from DecisionTree import train as DT

def getBestKValue(maxK, two_feature=False, features=['ecc', 'extent']):
    im=image.ProcessImages('train', two_value=two_feature, list_=features)
    X, Y=im.make_all()
    max = 0.0
    index = 3
    x_list = []
    y_list = []
    for x in range(3, maxK, 1):
        # for weight in ['uniform', 'distance']:
        print(' negighbours: ', 'distance', ' ', x)
        t2 = train_neigh(x, first_two=True, features=features)
        t2.train(X, Y)
        # t2.load(x, weight)
        v = t2.valid()
        if (v > max):
            max = v
            index = x
            print(v)
        x_list.append(x)
        y_list.append(v)

    plt.plot(x_list, y_list)
    plt.show()

    print('maximum: ', max, ' neighbours: ', index)
    return (max, index)

def train_neigh_():
    t = train_neigh(80)
    im = image.ProcessImages('train')
    X, Y = im.make_all()
    t.train_first_two_features(X, Y)
    max = 0.0
    index = 3
    x_list = []
    y_list = []
    # t3 = train_neigh(4)
    # print('test: ', t3.test())

    for x in range(3, 100, 1):
        # for weight in ['uniform', 'distance']:
        print(' negighbours: ', 'distance', ' ', x)
        t2 = train_neigh(x, first_two=True, features=['x0', 'y0'])
        #t2 = train_neigh(x, first_two=True)
        t2.train(X, Y)
        # t2.load(x, weight)
        v = t2.valid()
        if (v > max):
            max = v
            index = x
            print(v)
        x_list.append(x)
        y_list.append(v)

    plt.plot(x_list, y_list)
    plt.show()

    print('maximum: ', max, ' neighbours: ', index)


def main():
    #train_neigh_()
    getBestKValue(100, two_feature=True, features=['x0', 'y0'])
    tree=DT()
    im = image.ProcessImages('train')
    X, Y = im.make_all()
    tree.train(X, Y)
    print(tree.test())
    print(tree.valid())
    tree.show(X, Y)



    #out=measure.process('testscissors01-00.png')
    #tree.plot_tree(DT.fit(X, Y))
    #print(out2)
    #print(out)
    #t=train()
    #t.train(X, Y)
    #t.load()
    #print(t.predict(measure.process('test3.png')))
    #print(t.test())
    #cv2.showimage('h', measure.process('test3.png'))
    #cv2.waitKey(0)


    #t2=train_neigh()
    #t2.train(X, Y)
    #print(t2.predict(measure.process('test3.png')))
    #print(t2.test())





if __name__ == '__main__':
    main()