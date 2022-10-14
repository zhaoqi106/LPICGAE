import gensim
from Bio import SeqIO


def read_fa(path):
    res = {}
    rescords = list(SeqIO.parse(path, format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq)
        res[id] = seq
    return res


def train_doc2vec_model(seq_list, model_name):
    tokens = []
    for i, seq in enumerate(seq_list):
        items = []
        k = 0
        while k + 3 < len(seq):
            item = seq[k:k + 3]
            items.append(item)
            k = k + 1
        doc2vec_data = gensim.models.doc2vec.TaggedDocument(items, [i])
        tokens.append(doc2vec_data)
    print("-----begin train-----")

    model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=3, epochs=100, workers=12)
    model.build_vocab(tokens)
    model.train(tokens, total_examples=model.corpus_count, epochs=100)
    model.save(model_name + ".model")


if __name__ == '__main__':
    # lncrna_dict = read_fa("gencode.v39.lncRNA_translations.fa")
    # lncrna_list = list(lncrna_dict.values())
    # train_doc2vec_model(lncrna_list, "lncrna_doc2vec_test")
    protein_dict = read_fa("gencode.v39.pc_translations.fa")
    protein_list = list(protein_dict.values())
    train_doc2vec_model(protein_list, "protein_doc2vec_test")
