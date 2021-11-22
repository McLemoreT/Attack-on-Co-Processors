CALL activate JEC
cd C:\Users\Seth\Documents\GitHub\Attack-on-Co-Processors\

::@echo OFF

:: Define Path to Conda Installation and Environment Name so user input isn't needed
set CONDAPATH=C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)
set ENVNAME=JEC
:: Demo Flag. If "True", a single image will be tested for each non-ideality rather than the whole image set. 
set demo_mode = FALSE
PAUSE
:: Run a python script in that environment
::if %demo_mode% == TRUE (
::	python testing.py -d
::	python testing.py -dD
::	python testing.py -dE
::	python testing.py -dR
::	python testing.py -dF
::	python testing.py -dDE
::	python testing.py -dDR
::	python testing.py -dDF
::	python testing.py -dER
::	python testing.py -dEF
::	python testing.py -dRF
::	python testing.py -dDER
::	python testing.py -dDEF
::	python testing.py -dDRF
::	python testing.py -dERF
::	python testing.py -dDERF
::) else (
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
::)

:: Deactivate the environment
call conda deactivate 

PAUSE