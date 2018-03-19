import tensorflow as tf
import Lotto.lotto_datas as lotto_data
import Lotto.model as model
import numpy as np

def main():
    input_datas = lotto_data.getData('lotto_data.csv')
    output_datas = lotto_data.getOutputData('lotto_data.csv')

    print("Lenth:in:",len(input_datas),",out:",len(output_datas))
    len_of_data = 0
    if len(input_datas) > len(output_datas):
        len_of_data = len(output_datas)
    else:
        len_of_data = len(input_datas)
    print("Selected Length is ", len_of_data)

    '''
    # CNN Prediction Part
    with tf.Session() as sess:
        mainCNN = network.CNN(sess,input_size=7,output_size=45,name="mainCNN")

        tf.global_variables_initializer().run()
        
        for epoch in range(len_of_data):
            if epoch < len_of_data-200: # Learning Part
                i_data = np.array(input_datas[epoch])
                i_data = i_data.reshape(-1,7,7,1)
                o_data = np.array(output_datas[epoch])
                o_data = o_data.reshape(-1,45)
                #print("INPUT  : " , i_data)
                #print("OUTPUT : " , o_data)
                #print("---------------------------------")
                loss,_ = mainCNN.update(i_data,o_data)
                print("CNN Epoch : " , epoch , " / Loss : " , loss)
            else : # Testing Part
                i_data = np.array(input_datas[epoch])
                i_data = i_data.reshape(-1, 7, 7, 1)
                o_data = np.array(output_datas[epoch])
                o_data = o_data.reshape(-1, 45)
                pred = mainCNN.predict(i_data)
                #print(pred)
                for num in pred:
                    for _ in range(7):
                        max_val = num.argmax()
                        print(max_val+1)
                        num[max_val] = -999
                print("-------------------------------------")
    '''

    # RNN Prediction Part
    with tf.Session() as sess:
        mainRNN = model.RNN(sess,n_input=7,n_step=7,n_output=45,name="mainRNN")
        tf.global_variables_initializer().run()

        for epoch in range(len_of_data):
            if epoch < len_of_data-200: # Learning Part
                i_data = np.array(input_datas[epoch])
                i_data = i_data.reshape(-1, 7, 7)
                o_data = np.array(output_datas[epoch])
                o_data = o_data.reshape(-1, 45)
                loss, _ = mainRNN.update(i_data, o_data)
                print("RNN Epoch : ", epoch, " / Loss : ", loss)

            else: # Testing Part
                i_data = np.array(input_datas[epoch])
                i_data = i_data.reshape(-1, 7, 7)
                o_data = np.array(output_datas[epoch])
                o_data = o_data.reshape(-1, 45)
                pred = mainRNN.predict(i_data)
                # print(pred)
                for num in pred:
                    for _ in range(7):
                        max_val = num.argmax()
                        print(max_val + 1)
                        num[max_val] = -999
                print("-------------------------------------")

if __name__ == "__main__":
    main()