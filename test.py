import csv
def loadSampleData():
    with open('sampleAbstract.csv') as f:
        reader = csv.reader(f)
        abstract = list(reader)
    with open('sampleLabel.csv') as f:
        reader = csv.reader(f)
        label= list(reader)
    with open('sampleSignature.csv') as f:
        reader = csv.reader(f)
        signature = list(reader)
        signature = [e[0] for e in signature]
    return abstract, label, signature
