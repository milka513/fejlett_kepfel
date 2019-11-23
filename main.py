import  measure
import ProcessImage as image
from train import train

def main():
    #out=measure.process('testscissors01-00.png')
    im=image.ProcessImages('train')
    X, Y=im.make_all()
    #print(out2)
    #print(out)
    t=train()
    t.train(X, Y)
    #t.load()
    print(t.predict(measure.process('test1.png')))




if __name__ == '__main__':
    main()