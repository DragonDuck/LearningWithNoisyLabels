This experiment shows how deep neural networks (DNNs) perform on noisy
data. To this end, I trained a convolution DNN (structure:
conv-conv-pool-conv-conv-pool-dense) to classify MNIST handwritten
digits. However, the labels were randomly perturbed so that a defined
fraction of labels was guaranteed to be incorrect. After training, the
accuracy (F1-score) of the network was calculated *on the original,
unperturbed labels*. This mini-experiment shows that even after
significant scrambling, the true accuracy of the network remains nearly
unchanged.

    scrambling_results = read.csv(
      file = "ResultsScrambling_mnist_conv.csv", header = TRUE, row.names = NULL)

    ggplot(data = scrambling_results, mapping = aes(x = X, y = TrueF1)) + 
      geom_point(size = 4) + geom_line(size = 2) + custom_theme() + 
      coord_fixed() + labs(
        x = "Scrambling Factor", y = "F1-Score", 
        caption = paste0(
           "F1-Score of DNN trained on\n",
           "scrambled labels versus the\n",
           "true labels"))

![](README_files/figure-markdown_strict/unnamed-chunk-1-1.png)

    ggsave("ResultsScrambling_mnist_conv.pdf", width = 4, 
           height = 4, useDingbats = FALSE)

Next, I tested if iterative training, i.e. training a DNN on the outputs
of the previous iteration, showed any improvement. To this end, I
initially scrambled the labels so that 80% of them are incorrect
(iteration 0). The network was trained on the scrambled labels, its
accuracy evaluated, and the labels predicted for the entire dataset.
These new labels were used to train the next iteration of the DNN. There
seems to be no benefit to iterative training. While the accuracy still
increases slowly, the time cost of training over so many iterations far
outweighs the benefits of a minor increase in accuracy.

    iteration_results = read.csv(
      file = "ResultsIterative_mnist_conv_0.8.csv", 
      header = TRUE, row.names = NULL)
    iteration_results$X = as.factor(iteration_results$X)
    iteration_results$TextLoc = iteration_results$TrueF1 - 0.08
    iteration_results$Text = round(iteration_results$TrueF1, 2)

    ggplot(data = iteration_results, mapping = aes(x = X, y = TrueF1)) + 
      geom_col() + geom_text(mapping = aes(x = X, y = TextLoc, label = Text), 
                             color = "White", fontface = "bold", size = 3) +
      custom_theme() + ylim(c(0, 1)) + coord_flip() + 
      labs(
        x = "Iteration", y = "F1-Score", 
        caption = paste0(
           "F1-Score of DNN iteratively\n",
           "trained on its own output,\n",
           "starting with scrambled\n",
           "labels, versus the true labels"))

![](README_files/figure-markdown_strict/unnamed-chunk-2-1.png)

    ggsave("ResultsIterative_mnist_conv_0.8.pdf", width = 3, 
           height = 4, useDingbats = FALSE)

Testing this for a higher scrambling degree of 82.5% shows that
iteration still offers no benefit. However, the network saturates at a
lower accuracy.

    iteration_results = read.csv(
      file = "ResultsIterative_mnist_conv_0.825.csv", 
      header = TRUE, row.names = NULL)
    iteration_results$X = as.factor(iteration_results$X)
    iteration_results$TextLoc = iteration_results$TrueF1 - 0.08
    iteration_results$Text = round(iteration_results$TrueF1, 2)

    ggplot(data = iteration_results, mapping = aes(x = X, y = TrueF1)) + 
      geom_col() + geom_text(mapping = aes(x = X, y = TextLoc, label = Text), 
                             color = "White", fontface = "bold", size = 3) +
      custom_theme() + ylim(c(0, 1)) + coord_flip() + 
      labs(
        x = "Iteration", y = "F1-Score", 
        caption = paste0(
           "F1-Score of DNN iteratively trained on\n",
           "its own output starting with scrambled\n",
           "labels, versus the true labels"))

![](README_files/figure-markdown_strict/unnamed-chunk-3-1.png)

    ggsave("ResultsIterative_mnist_conv_0.825.pdf", width = 4, 
           height = 3, useDingbats = FALSE)
