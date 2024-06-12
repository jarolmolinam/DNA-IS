class conv_inputs:
    def __init__(self, file):
        self.filePath = file
        self.atomDicc = {
            "H": "1",
            "C": "6",
            "N": "7",
            "O": "8",
            "P": "15",
            "S": "16"
        }

    def title(self):
        tt = []
        with open(self.filePath, 'r') as fpdb:
            for line in fpdb:
                key = line.split()
                if len(key) == 11:
                    tipo = key[10]
                    nom = key[2]
                    tt.append(int(self.atomDicc.get(tipo, -1)))
        return tt


