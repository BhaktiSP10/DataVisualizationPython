import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings 
from statsmodels import robust

warnings.filterwarnings("ignore") 

dfh = pd.read_csv("haberman.csv")
print(dfh.count)
print(dfh.columns)
print(dfh["status"].value_counts()) #here 225 people survived more than 5 years and 81 survived less than 5 years
dfh.plot(kind='scatter', x='nodes', y='age') ;
plt.title("Scatter graph for Haberman Data Set using age and nodes")
plt.show()
dfh.plot(kind='scatter', x='age', y='year') ;
plt.title("Scatter graph for Haberman Data Set using age and year")
plt.show()
# Dataset understanging using visualization

#Pairplots to see data classification
sns.set_style("darkgrid");
sns.FacetGrid(dfh,hue="status",height=4) \
   .map(plt.scatter,"nodes","age") \
   #.add_legend();
plt.title("Scatter graph for Haberman Data Set using age and nodes")
plt.legend(["1", "2"], loc ="upper right") 
plt.show();
plt.close();

sns.set_style("whitegrid");
sns.pairplot(dfh, hue="status", height=3);
plt.suptitle("Pair plots for Status variation in haberman data")
plt.show()
#axil nodes and age can be used for analysis as it can give the status of survival as we can get some what pdf like structure from plot 3 and plot 7

surv_stat_1 = dfh.loc[dfh["status"] == 1];
surv_stat_2 = dfh.loc[dfh["status"] == 2];
#print(type(surv_stat_1))
plt.plot(surv_stat_1["nodes"],np.zeros_like(surv_stat_1['nodes']), 'o')
plt.plot(surv_stat_2["nodes"],np.zeros_like(surv_stat_2['nodes']), 'o')
plt.legend(["1","2"])
plt.title("Scatter plot")
plt.show()
#here individual status classification is extracted in different dfs and plotted in a scatter plot..where conclusion is difficult as both are overlapping

#PDF for the dataset
sns.FacetGrid(dfh, hue="status", height=6) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.title("PDF using status and age")
plt.show(); 
#overlapping status between age group of 35 to 75

sns.FacetGrid(dfh, hue="status", height=6) \
   .map(sns.distplot, "nodes") \
   .add_legend();
plt.title("PDF using status and nodes")
plt.show(); 
#here if nodes<=0 survival is more and if nodes>=0 and nodes<=3 survival can still be more
#if nodes>3 survival chances are less
#chances of error is more as we are not sure by PDFs


#CDF for the dataset
counts, bin_edges = np.histogram(surv_stat_1['nodes'], bins=10,density = True)

pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf) #calculates cdf
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(surv_stat_2['nodes'], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)

plt.legend(["1","2"])
plt.title("CDF using nodes")
plt.show();
#here yellow is with status 1 and red with status 2
#100% people have status 2 if nodes>4 99% if node=3(approx)
#81% people have status 1 i.e. more than 5ys of survival if node<1

#tried using age as a parameter but every value for analysis was nearly same eg: mean age was 52 for both


#Understanding the statistics
print("mean for survival:")
print(np.mean(dfh["nodes"]))
print(np.mean(surv_stat_1["nodes"]))
print(np.mean(surv_stat_2["nodes"]))
print(np.mean(np.append(surv_stat_1["nodes"],80))); 

#with outlier mean here is not much affected
#here mean of status 2 is higher , hence probability of less tha 5yrs of survival is more

print("\nStd-dev:");
print(np.std(surv_stat_1["nodes"]))
print(np.std(surv_stat_2["nodes"]))
#sd for status 2 is more , hence spread for status 2 is more
print("median for survival:")
print(np.median(dfh["nodes"]))
print(np.median(surv_stat_1["nodes"]))
print(np.median(surv_stat_2["nodes"]))
print(np.median(np.append(surv_stat_1["nodes"],80)));
# people with 0 nodes has more chances of survival and less chances if people with 4 or more nodes


print("\nQuantiles:")
print(np.percentile(surv_stat_1["nodes"],np.arange(0, 100, 25))) #25% people have more than 3 nodes
print(np.percentile(surv_stat_2["nodes"],np.arange(0, 100, 25))) #25% people have more than 11 nodes

print("\n90th Percentiles:")
print(np.percentile(surv_stat_1["nodes"],90)) #max node is 8 for people surviving more than 5 yrs
print(np.percentile(surv_stat_2["nodes"],90)) #max node is 20 for people not surviving more than 5 yrs


print ("\nMedian Absolute Deviation")
print(robust.mad(surv_stat_1["nodes"]))
print(robust.mad(surv_stat_2["nodes"]))
# people with 0 nodes have more chances of survival and less chances if people with more than 5 axil nodes


#Box Plot representation
sns.boxplot(x='status',y='nodes', data=dfh)
plt.legend(["1","2"])
plt.title("Box plot using status and nodes")
plt.show()
#25th percentile is 0 and 75th is nearlly 4 and median is 0 for status 1
#25th percentile is 1 and 75th is nearlly 11 and median is 4 for status 2

#Violin Plot representation
sns.violinplot(x="status", y="nodes", data=dfh, size=8)
plt.legend(["1","2"])
plt.title("Violin plot using status and nodes")
plt.show()

#Joint Plot representation
sns.jointplot(x="nodes",y="age",data=surv_stat_1,kind="kde")
plt.title("Contour plot using age and nodes")
plt.show()


"""Hence, we can say that people with 0-3 auxilary nodes can survive greater than 5yrs 
and people with more than 4-11 auxilary nodes
can survive less than 5yrs ...based on the above analysis..some can be exceptions"""

