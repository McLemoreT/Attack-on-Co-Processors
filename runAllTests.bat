CALL activate JEC
cd C:\Users\Seth\Documents\GitHub\Attack-on-Co-Processors\

::@echo OFF

:: Define Path to Conda Installation and Environment Name so user input isn't needed
set CONDAPATH=C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)
set ENVNAME=JEC

:: Run a python script in that environment
python testing.py 
python testing.py -D
python testing.py -E
python testing.py -R
python testing.py -F
python testing.py -DE
python testing.py -DR
python testing.py -DF
python testing.py -ER
python testing.py -EF
python testing.py -RF
python testing.py -DER
python testing.py -DEF
python testing.py -DRF
python testing.py -ERF
python testing.py -DERF

:: Deactivate the environment
call conda deactivate 

PAUSE