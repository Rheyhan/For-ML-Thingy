from Modules.advanced_epochandbias import NeuralNetwork

class brute:
    def __init__(self, no_of_in_nodes, no_of_out_nodes, train_imgs, train_labels, test_imgs, test_labels, train_labels_one_hot):
        '''Input the data you want to brute'''
        self.no_of_in_nodes=no_of_in_nodes
        self.no_of_out_nodes=no_of_out_nodes
        self.train_imgs=train_imgs
        self.train_labels=train_labels
        self.test_imgs=test_imgs
        self.test_labels=test_labels
        self.train_labels_one_hot=train_labels_one_hot
        
    def start(self, path="Noname.csv", hidden_nodes=[20,40,60,80,100], learning_rates=[0.01, 0.05], biases= [None, 0.5], epochs=5):
        '''Configuration for brute, the more input will take more time'''
        with open(path, "w") as fh_out:
                for hidden_node in hidden_nodes:
                    for learning_rate in learning_rates:
                        for bias in biases:
                            print(f'BEGINNNING  |hidden_nodes: {hidden_node}    |learning_rate: {learning_rate}    |bias:  {bias}')
                            ANN = NeuralNetwork(no_of_in_nodes=self.no_of_in_nodes, no_of_out_nodes=self.no_of_out_nodes,
                                                    no_of_hidden_nodes=hidden_node,
                                                    learning_rate=learning_rate,
                                                    bias=bias)
                            weights = ANN.train(self.train_imgs, self.train_labels_one_hot, epochs=epochs, intermediate_results=True)
                            for epoch in range(epochs):
                                print("|", end="")
                                ANN.wih = weights[epoch][0]
                                ANN.who = weights[epoch][1]
                                train_corrects, train_wrongs = ANN.evaluate(self.train_imgs, self.train_labels)
                                test_corrects, test_wrongs = ANN.evaluate(self.test_imgs,self.test_labels)
                                outstr = str(hidden_node) + " " + str(learning_rate) + " " + str(bias)
                                outstr += " " + str(epoch) + " "
                                outstr += str(train_corrects / (train_corrects + train_wrongs)) + " "
                                outstr += str(train_wrongs / (train_corrects + train_wrongs)) + " "
                                outstr += str(test_corrects / (test_corrects + test_wrongs)) + " "
                                outstr += str(test_wrongs / (test_corrects + test_wrongs))
                                fh_out.write(outstr + "\n" )
                                fh_out.flush()
                            print("\n")

