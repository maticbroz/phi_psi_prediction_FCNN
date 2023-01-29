import urllib.request
import os

def download_pdb(pdbcode, datadir, downloadurl="http://ufq.unq.edu.ar/codnas/pdbs_codnas/"):
    pdbfn = pdbcode.strip()
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(f"Error: {err}")
        return None

def main():
    with open("../Inputs/Codnas_list.txt") as f:
        for pdb in f:
            download_pdb(pdb.strip(), "Codnas_PDBs/", "http://ufq.unq.edu.ar/codnas/pdbs_codnas/")

if __name__ == "__main__":
    main()
