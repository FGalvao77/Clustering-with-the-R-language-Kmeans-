#### SEGMENTAÇÃO DE CLIENTES [Clustering w/ R] ####

#######  Visualizando a área de trabalho do R #######
getwd()

### Instalando os pacotes iniciais ###
install.packages('tidyverse')
install.packages('cluster')
install.packages('factoextra')

# ## se por acaso, tiver algum erro no pacote "factoextra" utilize os comandos abaixo ##
# sudo apt-get install cmake
# install.packages('factoextra', dependencies = TRUE)

# install.packages('devtools')
# devtools::install_github('kassambara/factoextra')

## habilitando os pacotes para uso ##
library(tidyverse)
library(cluster)
library(factoextra)

# consultando as funções presente no pacote "tidyverse" ##
print(tidyverse_packages())

### Importando e explorando o conjunto de dados ###

## carregando o conjunto de dados direto de uma "url" ##
bdados <- read_csv('https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv')

## tipo do objeto ##
print(class(bdados))

## nome das colunas ##
print(names(bdados))  

## estrutura do objeto ##
print(str(bdados))    

## visualizando as 6 primeiras observações ##
print(head(bdados)) 

## visualizando as 6 últimas observações ##
print(tail(bdados))  

## dimensão do conjunto de dados ##
print(dim(bdados))  # linhas e colunas

## # estatística descritiva do conjunto de dados ##
print(summary(bdados))

### Visualização gráfica ###

ggplot(data=bdados, aes(x=Gender)) +
  geom_bar(stat='count', fill='darkgray', colour = "#3366FF") +
  xlab('')+
  theme_minimal() +
  labs(title = 'Distribuição do Sexo') +
  coord_flip()

ggplot(bdados, aes(x=Age)) + 
  geom_histogram(fill='darkgray', colour = "#3366FF", binwidth=1) +
  labs(title = 'Distribuição da Idades')

hist <- ggplot(bdados, aes(x=Age)) + 
  geom_histogram(fill='darkgray', colour = "#3366FF", binwidth=1)

hist + geom_vline(aes(xintercept=mean(Age)),
                  color='green', linetype="dashed", size=1) + 
  labs(title = 'Distribuição da Idade [linha da média]')

ggplot(bdados, aes(x=Age)) + 
  geom_histogram(aes(y=..density..), fill='darkgray', colour = "#3366FF", binwidth=1)+
  geom_density(alpha=.2, fill='#FF6666') + 
  labs(title = 'Distribuição da Idades [densidade]')

hist <- ggplot(bdados, aes(x=bdados$`Annual Income (k$)`)) + 
  geom_histogram(fill='darkgray', colour = "#3366FF", binwidth=1,) + 
  coord_flip()
hist + geom_vline(aes(xintercept=mean(bdados$`Annual Income (k$)`)),
                  color='green', linetype="dashed", size=1) + 
  labs(title = 'Distribuição da Renda Anual [linha da média]')

ggplot(bdados, aes(x=bdados$`Annual Income (k$)`)) + 
  geom_histogram(aes(y=..density..), fill='darkgray', colour = "#3366FF", binwidth=1)+
  geom_density(alpha=.2, fill='#FF6666') + 
  labs(title = 'Distribuição da Renda Anual [densidade]')

### Tratamento dos dados ###

## vamos usar a função dummyVars() do pacote "caret" para executar a codificação one-hot aa variável "Gender" (categórica) ##
install.packages('caret') # instalando o pacote
library(caret)            # habilitando o pacote

## selecionando a coluna de interesse e salvando-a em um novo objeto ##
dummy_gender <- bdados[, ('Gender')]

## visualizando a nova variável ##
print(head(dummy_gender))

## aplicando a função "dummyVars" no objeto "dummy_gender" e salvando o seu resultado na variável "dummy" ##
dummy <- dummyVars('~.', data=dummy_gender)

## realizando as predições com o obejto "dummy" na base de dados original e já transformando em um data frame ##
dummies_gender <- data.frame(predict(dummy, newdata=bdados))

## visualizando o nome da novas colunas criadas pelo "dummy" ##
names(dummies_gender)

## visualizando o conteúdo do objeto ##
print(head(dummies_gender))

## realizando a seleção das vars do conjunto de dados original e concatenando com o "dummies_gender" e,
## salvando o resultado no objeto "df_final" ##
df_final <- bind_cols(bdados[, c('Age', 'Annual Income (k$)', 
                                 'Spending Score (1-100)')], dummies_gender)

## visualizando as 6 primeiras observações do novo objeto ##
print(head(df_final))

## visualizando a dimensão do novo objeto ##
print(dim(df_final))    # linhas e colunas

## estatística descritiva do conjunto de dados final ##
print(summary(df_final))

## omitindo dados faltantes ##
df_final <- na.omit(df_final)

### Visualização gráfica ##

## com a função "fviz_nbclust" podemos visualizar o número de clusters "ideal", técnica conhecida como "curva do cotovelo" ##
fviz_nbclust(df_final, kmeans, method='gap_stat')

## utilizando a função "clusGap" para instanciar nosso objeto final, "df_final" e
## definindo alguns parâmetros essenciais ##
gap_statistic <- clusGap(df_final,
                         FUN=kmeans,
                         nstart=20,
                         K.max=10,
                         B=100)

## aplicando objeto na função "fviz_gap_stat" e visualizando a "curva do cotovelo" ##
fviz_gap_stat(gap_statistic)

## aplicando o "kmeans" ##
set.seed(2022)  # setando a semente aleatória
km <- kmeans(df_final, centers=2, nstart=25)
km  # visualizando o resultado 

## com a função "fviz_cluster" visualizando a silhueta dos clusters ##
fviz_cluster(km, data=df_final)

## visualizando a distribuição dos clusters ##
aggregate(df_final, by=list(cluster=km$cluster), mean)

## concatenando o número do cluster no conjunto de dados original ##
final_data <- cbind(bdados, cluster = km$cluster)

## visualizando as 15 primeiras observações ##
head(final_data, 15)

## visualizando as 15 últimas observações ##
tail(final_data, 15)

## visualizando o sumário do cluster 1 ##
summary(final_data %>% filter(cluster == 1))

## visualizando o sumário do cluster 2 ##
summary(final_data %>% filter(cluster == 2))

## distância euclidiana ##
dist <- dist(df_final[ , c(1:5)] , diag=TRUE)
hc <- hclust(dist) # clustering hierárquico com hclust
plot(hc) # plot do resultado

