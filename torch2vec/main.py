from utils import *
from SkipW2V import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # File I/O options
    parser.add_argument("-c", "--corpus_path", type=str, default="../data/1bwc.txt", help="Path to the 1bwc corpus")
    parser.add_argument("-w", "--window", type=int, default=2, help="Number of words to take front and back from the center word") # so default is a (2*2)+1 sliding window
    parser.add_argument("-min", "--min_count", type=int, default=0, help="Minimum required count of occurence")
    parser.add_argument("-ll", "--line_limit", type=int, default=100, help="Number of lines to read from the corpus")

    # Negative Sampling options
    parser.add_argument("-tsize", "--table_size", type=int, default=10000, help="Size of the unigram table")
    parser.add_argument("-nex", "--number_neg_examples", type=int, default=5, help="Number of added negative examples")

    # W2V options
    parser.add_argument("-dim", "--embed_dim", type=int, default=128, help="Size of the embedding")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-opt", "--optimizer", type=str, default="sgd", help="Optimizer used")
    parser.add_argument("-batch", "--batch_size", type=int, default=200, help="Size of batches used for training")
    parser.add_argument("-e", "--epochs", type=int, default=int, help="Number of epochs")

    # Train/test mode
    parser.add_argument("-train", "--train_test", type=str, default="train", help="Whether to run in train or test mode")
    parser.add_argument("-words", "--test_words", nargs="+", type=str)

    args = parser.parse_args()

    if args.train_test is "train":
        # Reading data
        sg_data, intw, wint, occ_dict, excluded_words, total_word_count, true_total = read_data(args.corpus_path,
                                                                                                window=args.window,
                                                                                                mincount=args.min_count,
                                                                                                line_limit=args.line_limit)
        unigrams_table = build_unigram_table(occ_dict, table_size=args.table_size)
        sg_neg_examples_data = add_negative_samples(sg_data, unigrams_table, neg_examples_size=args.number_neg_examples)

        # Parameterizing
        n_unique_words = len(intw)
        w2v = SkipW2V(n_unique_words, args.embed_dim)
        if args.optimizer == "sgd":
            opt = optim.SGD(w2v.parameters(), lr=args.learning_rate)
        elif args.optimizer == "adam":
            opt = optim.Adam(w2v.parameters(), lr=args.learning_rate)

        # Training
        n_batches = int(len(sg_neg_examples_data)/args.batch_size)
        for e in range(args.epochs):
            if e is 0: desc = "Epoch - 0"
            else: desc = "Epoch - {}, [Epoch - {} loss = {}]".format(e, (e-1), str(float(mean_loss))[:5])
            for i in trange(n_batches, desc=desc, unit=" batches"):
                # Editing progress bar
                try:
                    batch = sg_neg_examples_data[(i*args.batch_size):((i*args.batch_size)+args.batch_size)]
                except IndexError:
                    batch = sg_neg_examples_data[(i*args.batch_size):]
                w2v.zero_grad()
                mean_loss = w2v(batch)
                mean_loss.backward()
                opt.step()

        print("\nDone Training. Saving model to ./model/model.pt")
        torch.save(w2v, "./model/model.pt")
        _ = save_dict("./model/w2int.pkl", wint)
        _ = save_dict("./model/int2w.pkl", intw)

    else:
        print("\nLoading model, w-2-int and int-2-w dict at ./model/\n")
        w2v = torch.load("./model/model.pt")
        wint = load_dict("./model/w2int.pkl")
        intw =load_dict("./model/int2w.pkl")

        for w in args.test_words:
            print("Nearest words to {}:\n".format(w))
            for result in nearest_words(w2v.first_layer, len(wint), w, wint, intw, n_words=10):
                print(
                        tuple([
                                str(result[0])[:5], result[1]
                                ])
                        )
        print("\n")
