import sys
sys.dont_write_bytecode = True

from essential_methods import *

# Execução do dataset Student Performance 
caminho_arquivo1 = 'datasets/Student_Performance.csv' 
variaveis_independentes1 = ['Hours Studied','Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
variavel_dependente1 = 'Performance Index'
executar_analise(caminho_arquivo1, variaveis_independentes1, variavel_dependente1)

# Execução do dataset Student's math score for different teaching style
caminho_arquivo2 = 'datasets/1ResearchProjectData.sav' 
variaveis_independentes2 = ['Student', 'Teacher', 'Gender', 'Ethnic', 'Freeredu', 'wesson']
variavel_dependente2 = 'Score'
executar_analise(caminho_arquivo2, variaveis_independentes2, variavel_dependente2)

# Execução do dataset Student Marks
caminho_arquivo3 = 'datasets/Student_Marks.csv' 
variaveis_independentes3 = ['number_courses','time_study']
variavel_dependente3 = 'Marks'
executar_analise(caminho_arquivo3, variaveis_independentes3, variavel_dependente3)