import glob

def get_coordinates(line):
    return line[30:38].strip() + "," + line[38:46].strip() + "," + line[46:54].strip()

def main():
    prev_atom = "C"
    output_file = "Codnas_coordinates.csv"
    header = "nr,Res,Nx,Ny,Nz,Cax,Cay,Caz,Cx,Cy,Cz\n"

    with open(output_file, "a") as f:
        f.write(header)
        
        for pdb_file in glob.iglob("Codnas_PDBs/*"):
            with open(pdb_file, "r") as pdb:
                current_index = 1
                
                for line in pdb:
                    if line.startswith("ATOM"):
                        if current_index < int(line[7:11].strip()):
                            current_index = int(line[7:11].strip())

                            if line[12:16].strip() == "N" and prev_atom == "C":
                                output = line[22:26].strip() + "," + line[17:20].strip() + "," + get_coordinates(line) + ","
                                f.write(output)
                                prev_atom = "N"    
                                
                            elif line[12:16].strip() == "CA" and prev_atom == "N":
                                output = get_coordinates(line) + ","
                                f.write(output)  
                                prev_atom = "CA"        

                            elif line[12:16].strip() == "C" and prev_atom == "CA":
                                output = get_coordinates(line) + "\n"
                                f.write(output)    
                                prev_atom = "C"  

if __name__ == '__main__':
    main()
