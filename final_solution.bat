@echo off
echo Removing LFS pointer file...
del Scripts\debug_sample.csv
echo.
echo Rebuilding repository...
rd /s /q .git
git init
git branch -M main
git add .
git commit -m "Initial commit: Federated Edge Intelligence System"
git remote add origin https://github.com/Themaximum929/Traffic-Optimization-using-RL.git
git push -u origin main
echo.
echo Success!
pause
