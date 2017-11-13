Book Recommendation System

Download Dataset from  https://drive.google.com/open?id=1MPc--TuBaxoI9v31n-Acg9PtsrReERle and put the two files in a folder named Dataset.

for a category named cateogryname:
    a. to get books of the cateogory run ---->   python getBooksbyCategory categoryname
    b. to form the graphs for the category run ---->  python getGraphs.py categoryname
    c. for svm model run ---> python svm.py categoryname
    d. for rnn model run ---> python rnn.py categoryname
    e. for deewalk feature with svm model run 
        1. Bash command ----> deepwalk --input ./Graphs/categoryname/item_item.csv --output ./Embeddings/categoryname/items.embeddings --format edgelist --representation-size=100
        2. then run ----> python deepwalkmodel.py categoryname
    d. for neural network model run ---> python nn.py categoryname
two category names are valid:  1. "Publishing & Books"   and 2. "Finance & Investing"



Presentation Link can be found here : https://drive.google.com/open?id=1Z0eG9UL6OoPqL3QMyRIl4yGC1JH8E13bpNLPtwGKXlk
Video explaining the project can be found here https://drive.google.com/open?id=1TDsxpfEEqneUb7aHbF6taM-5G3Qd2gwT


