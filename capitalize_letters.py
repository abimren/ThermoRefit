"""
Capitalize all letters
"""
with open("therm.dat", "r") as inputFile:
  content = inputFile.read()
with open("upper_therm.dat", "w") as outputFile:
  outputFile.write(content.upper())
with open("chem.inp", "r") as inputFile:
  content = inputFile.read()
with open("upper_chem.inp", "w") as outputFile:
  outputFile.write(content.upper())  
