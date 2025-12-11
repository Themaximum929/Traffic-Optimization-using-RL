@echo off
echo Completely rebuilding Git repository...
echo.
rd /s /q .git
git init
git branch -M main
git add .
git commit -m "Initial commit: Federated Edge Intelligence System"
git remote add origin https://github.com/Themaximum929/Traffic-Optimization-using-RL.git
git push -u origin main
echo.
echo Done!
pause
