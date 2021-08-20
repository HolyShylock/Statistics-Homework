library(ggplot2)

data<-read.table("C:/Users/86182/Documents/WeChat Files/wxid_pa2858gtxhb622/FileStorage/File/2021-08/heart_failure_clinical_records_dataset.csv",header=TRUE,sep=",")
data[,13]<-factor(data[,13])
i=12
ggplot(data,aes(x=data[,i],fill=DEATH_EVENT))+geom_histogram(position = "identity")+scale_x_continuous(name=colnames(data)[i])

ggplot(data,aes(x=DEATH_EVENT))+geom_histogram(stat="count")+labs(x="DEATH_EVENT")

