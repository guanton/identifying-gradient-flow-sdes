rm(list = ls())
gc()

cor_dat <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/metric=grid_cos_p0-gmm_pointwise.csv")
mae_dat <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/metric=grid_mae_p0-gmm_pointwise.csv")

cor_dat$x<-round(cor_dat$x,2)
cor_dat$y<-round(cor_dat$y,2)

mae_dat$x<-round(mae_dat$x,2)
mae_dat$y<-round(mae_dat$y,2)

t.test(x=mae_dat$value[mae_dat$method=="wot" & mae_dat$potential=="poly"],mae_dat$value[mae_dat$method=="appex" & mae_dat$potential=="poly"],paired=T)

mae_dat_agg<-mae_dat[!duplicated(mae_dat[,c("potential","method","seed")]),]
cor_dat_agg<-cor_dat[!duplicated(cor_dat[,c("potential","method","seed")]),]

for(i in 1:nrow(mae_dat_agg)){
  mae_dat_agg$value[i]<-mean(mae_dat$value[mae_dat$potential==mae_dat_agg$potential[i] & mae_dat$method==mae_dat_agg$method[i] & mae_dat$seed==mae_dat_agg$seed[i]])
  cor_dat_agg$value[i]<-mean(cor_dat$value[cor_dat$potential==cor_dat_agg$potential[i] & cor_dat$method==cor_dat_agg$method[i] & cor_dat$seed==cor_dat_agg$seed[i]])
}


t.test(x=mae_dat_agg$value[mae_dat_agg$method=="wot" & mae_dat_agg$potential=="poly"],
       mae_dat_agg$value[mae_dat_agg$method=="appex" & mae_dat_agg$potential=="poly"],paired=T)

t.test(x=mae_dat_agg$value[mae_dat_agg$method=="MIT" & mae_dat_agg$potential=="poly"],
       mae_dat_agg$value[mae_dat_agg$method=="appex" & mae_dat_agg$potential=="poly"],paired=T)

wilcox.test(x=mae_dat_agg$value[mae_dat_agg$method=="MIT"],
            mae_dat_agg$value[mae_dat_agg$method=="appex"],paired=T)

wilcox.test(x=mae_dat_agg$value[mae_dat_agg$method=="jkonet_star"],
            mae_dat_agg$value[mae_dat_agg$method=="appex"],paired=T)


wilcox.test(x=cor_dat_agg$value[cor_dat_agg$method=="MIT"],
            cor_dat_agg$value[cor_dat_agg$method=="appex"],paired=T)

wilcox.test(x=cor_dat_agg$value[cor_dat_agg$method=="jkonet_star"],
            cor_dat_agg$value[cor_dat_agg$method=="appex"],paired=T)


# separate results by drift magnitude
#drift_magnitude_quadratic <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/drift_magnitude_quadratic.csv")
#drift_magnitude_quadratic <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/drift_magnitude_wavy_plateau.csv")
#drift_magnitude_quadratic <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/drift_magnitude_styblinski_tang.csv")
#drift_magnitude_quadratic <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/drift_magnitude_oakley_ohagan.csv")
drift_magnitude_quadratic <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/drift_magnitude_bohachevsky.csv")
curpot<-"bohachevsky"
drift_magnitude_quadratic$x<-round(drift_magnitude_quadratic$x,2)
drift_magnitude_quadratic$y<-round(drift_magnitude_quadratic$y,2)
mae_dat_qu<-mae_dat[mae_dat$potential==curpot,]
mae_dat_qu<-merge(mae_dat_qu,drift_magnitude_quadratic,by=c("x","y"),all.x=T)

cor_dat_qu<-cor_dat[cor_dat$potential==curpot,]
cor_dat_qu<-merge(cor_dat_qu,drift_magnitude_quadratic,by=c("x","y"),all.x=T)


appex_mae<-mean(mae_dat_qu$value[mae_dat_qu$method=="appex" & mae_dat_qu$magnitude< median(mae_dat_qu$magnitude)])
sbirr_mae<-mean(mae_dat_qu$value[mae_dat_qu$method=="MIT" & mae_dat_qu$magnitude< median(mae_dat_qu$magnitude)])

paste("MAE diff at low potential zones is: " ,sbirr_mae-appex_mae)


appex_mae<-mean(mae_dat_qu$value[mae_dat_qu$method=="appex" & mae_dat_qu$magnitude> median(mae_dat_qu$magnitude)])
sbirr_mae<-mean(mae_dat_qu$value[mae_dat_qu$method=="MIT" & mae_dat_qu$magnitude> median(mae_dat_qu$magnitude)])

paste("MAE diff at high potential zones is: " ,sbirr_mae-appex_mae)


appex_cor<-mean(cor_dat_qu$value[cor_dat_qu$method=="appex" & cor_dat_qu$magnitude< median(cor_dat_qu$magnitude)])
sbirr_cor<-mean(cor_dat_qu$value[cor_dat_qu$method=="MIT" & cor_dat_qu$magnitude< median(cor_dat_qu$magnitude)])

paste("cor diff at low potential zones is: " ,appex_cor- sbirr_cor)

appex_cor<-mean(cor_dat_qu$value[cor_dat_qu$method=="appex" & cor_dat_qu$magnitude> median(cor_dat_qu$magnitude)])
sbirr_cor<-mean(cor_dat_qu$value[cor_dat_qu$method=="MIT" & cor_dat_qu$magnitude> median(cor_dat_qu$magnitude)])

paste("cor diff at high potential zones is: " ,appex_cor- sbirr_cor)




############################################################################################################################################################
###################################################3   run appex on the quadratic example included in the nn-APPEX paper   #################################
############################################################################################################################################################

rm(list=ls())
generate_data<-function(A,G,T_max, traj, dt,dt_EM,kill=FALSE, X0dist="const",X0mean=0,X0sd=0){
  dim= sqrt(length(A))
  N_max=round(T_max/dt_EM)
  X_vec=array(0, dim=c(N_max, dim, traj))
  
  if(kill){
    X_vec_cf=array(0, dim=c(N_max, dim, traj*N_max))
    W_vec_cf=X_vec_cf
    W_vec_cf[1,,]<- 0
    if(X0dist== "multi"){
      whichX0col<-sample(1:ncol(X0mean),traj*N_max,replace = T)
      for(d in 1:dim){
        X_vec_cf[1,d,]<-X0mean[d,whichX0col]
      }
    }
    if(X0dist=="const") X_vec_cf[1,,]<-X0mean
    if(X0dist=="normal") X_vec_cf[1,,]<- rnorm(dim*traj*N_max,mean = X0mean,sd=X0sd)
    if(X0dist=="unif") X_vec_cf[1,,]<- runif(dim*traj*N_max,min = X0mean,max=X0sd)
    for(i in 2:N_max){
      W_vec_cf[i,,]= W_vec_cf[i-1,,]+rnorm(dim*traj*N_max,mean = 0,sd=sqrt(dt_EM))
      X_vec_cf[i,,]= X_vec_cf[i-1,,] + A%*%X_vec_cf[i-1,,]*dt_EM + G%*%(W_vec_cf[i,,]- W_vec_cf[i-1,,])
    }
    
    counter=1
    for(i in 1:N_max){
      for(j in 1:traj){
        X_vec[i,,j]= X_vec_cf[i,,counter]
        counter=counter+1
      }
    }
    return(list(givenData=X_vec[seq(from=1,by=round(dt/dt_EM),length.out=round(T_max/dt)),,,drop=F],Counterfactuals=X_vec_cf))
  } else{
    W_vec=X_vec
    W_vec[1,,]<- 0
    if(X0dist== "multi"){
      whichX0col<-sample(1:ncol(X0mean),traj,replace = T)
      for(d in 1:dim){
        X_vec[1,d,]<-X0mean[d,whichX0col]
      }
    }
    if(X0dist=="const") X_vec[1,,]<-X0mean
    if(X0dist=="normal") X_vec[1,,]<- rnorm(dim*traj,mean = X0mean,sd=X0sd)
    if(X0dist=="unif") X_vec[1,,]<- runif(dim*traj,min = X0mean,max=X0sd)
    for(i in 2:N_max){
      W_vec[i,,]= W_vec[i-1,,]+rnorm(dim*traj,mean = 0,sd=sqrt(dt_EM))
      X_vec[i,,]= X_vec[i-1,,] + A%*%X_vec[i-1,,]*dt_EM + G%*%(W_vec[i,,]- W_vec[i-1,,])
    }
    return(list(givenData=X_vec[seq(from=1,by=round(dt/dt_EM),length.out=round(T_max/dt)),,,drop=F],Counterfactuals=X_vec))
  }
  
}

library(mvnfast)
library(transport)
library(Rfast)
library(T4transport)
library(pracma)
library(Matrix)
library(foreach)
library(doParallel)
library(mvtnorm)

mysinkho2<-function(K, maxiter,u_thresh){
  a=matrix(rep(1/nrow(K),nrow(K)),ncol = 1)
  b=matrix(rep(1/ncol(K),ncol(K)),ncol=1)
  u=matrix(rep(1,nrow(K)),ncol = 1)
  v=matrix(rep(1,ncol(K)),ncol=1)
  i_in_sink=1
  diff<-1
  u_thresh<-u_thresh
  cur_out<-diag(as.numeric(u))%*%K%*%diag(as.numeric(v))
  while(i_in_sink<maxiter & diff>u_thresh){
    cur_out_old<-cur_out
    u= a/(K%*%v)
    v = b/(t(K)%*%u) # t(G)
    cur_out= diag(as.numeric(u))%*%K%*%diag(as.numeric(v))
    diff<-max(abs(cur_out_old - cur_out))
    i_in_sink=i_in_sink+1
  }
  #print(i_in_sink)
  return(cur_out)
}

mysinkho<-function(cost, lam, maxiter){
  G = exp(-cost/lam)
  a=matrix(rep(1/nrow(cost),nrow(cost)),ncol = 1)
  b=matrix(rep(1/ncol(cost),ncol(cost)),ncol=1)
  uold=matrix(rep(1,nrow(cost)),ncol = 1)
  vold=matrix(rep(1,ncol(cost)),ncol=1)
  for (i in 1:maxiter){
    unew = a/(G%*%vold)
    vnew = b/(t(G)%*%unew) # t(G)
    uold     = unew
    vold     = vnew
  }
  return(diag(as.numeric(unew))%*%G%*%diag(as.numeric(vnew)))
}


Alist=list(matrix(c(-10,0,0,-10),nrow=2))
Glist=list(matrix(c(sqrt(0.2),0,0,sqrt(0.2)),nrow=2))
traj=2000
N_max=3
dt=0.01
dt_EM=0.01
Tmax=dt*N_max

iter=30
rep=3
multi_sinkho_A<-array(0,dim = c(rep,iter,length(Alist)))
multi_sinkho_G<-array(0,dim= c(rep,iter,length(Alist)))



for(s in 1:length(Alist)){
  
  A=Alist[[s]]
  G=Glist[[s]]
  dim=nrow(A)
  dat<-generate_data(A,G,T_max=dt*N_max,traj = traj,dt_EM=dt_EM,dt=dt,kill = F,X0dist = "unif",X0mean = -4,X0sd = 4)
  
  for(r in 1:rep){
    A_est= -diag(dim)*0
    t_meanG<-mean(diag(G%*%t(G)))
    GGT_est=diag(dim)*runif(1,min=t_meanG*0.1,max=t_meanG*10)
    
    for(it in 1:iter){
      #GGT_est= G%*%t(G)*1.06
      #A_est= A
      X_vec_OT=dat$givenData
      
      tran_list<-list()
      linest<-(diag(dim)+A_est*dt)
      inv_cov<- -0.5*pinv(GGT_est*dt)
      K=matrix(0,nrow=traj,ncol=traj)
      time<-Sys.time()
      registerDoParallel(cl <- makeCluster(1))
      tran_list=foreach(t=2:N_max) %do% {
        for(i in 1:nrow(K)){
          dX=matrix(X_vec_OT[t,,],nrow = dim)- as.numeric(linest %*%matrix(X_vec_OT[t-1,,i],ncol = 1))
          Ki1<-rowSums((t(dX)%*%inv_cov)*t(dX))
          Ki<-exp(Ki1)
          Ki=dmvnorm(t(matrix(X_vec_OT[t,,],nrow = dim)),mean = as.numeric(linest %*%X_vec_OT[t-1,,i]),sigma = GGT_est*dt)
          if(sum(Ki)==0) GGT_est<-GGT_est+diag(dim)*1e-8
          Ki=dmvnorm(t(matrix(X_vec_OT[t,,],nrow = dim)),mean = as.numeric(linest %*%X_vec_OT[t-1,,i]),sigma = GGT_est*dt)
          K[i,]<-Ki
        }
        mysinkho2(K,maxiter = 50,u_thresh=1e-5)*traj
      }
      stopCluster(cl)
      Sys.time()- time
      
      # tran_list<-list()
      #    linest<-(diag(dim)+A_est*dt)
      #    inv_cov<- -0.5*pinv(GGT_est*dt)
      #    time<-Sys.time()
      #    for(t in 2:N_max){
      #        K=matrix(0,nrow=traj,ncol=traj)
      #        for(i in 1:nrow(K)){
      #            for(j in 1:ncol(K)){
      #               dX=matrix(X_vec_OT[t,,j]- linest %*%X_vec_OT[t-1,,i],ncol=1)
      #               K[i,j]=max(exp(t(dX)%*%inv_cov%*%dX),.Machine$double.xmin)
      #              }
      #          }
      #        tran_list[[t-1]]<-mysinkho2(K,maxiter = 2000)*traj
      #    }
      #    Sys.time()-time
      
      n_sim_traj=5*traj #create ROT trajectories
      pos_vec_ROT<-array(0,dim=c(N_max, dim, n_sim_traj))
      X_vec_ROT<-array(0,dim=c(N_max, dim, n_sim_traj))
      for(i in 1:dim(pos_vec_ROT)[3]){
        for(j in 1:dim(pos_vec_ROT)[1]){
          if(j==1){
            pos_vec_ROT[j,,i]<-sample(1:traj,1)
            X_vec_ROT[j,,i]<-X_vec_OT[j,,pos_vec_ROT[j,1,i]]
          } else {
            pos_vec_ROT[j,,i]<-sample(1:traj,1,prob = tran_list[[j-1]][pos_vec_ROT[j-1,1,i],] )
            X_vec_ROT[j,,i]<-X_vec_OT[j,,pos_vec_ROT[j,1,i]]
          }
        }
      }
      
      #X_vec_ROT<-X_vec_OT
      
      dX_vec_ROT= X_vec_ROT[2:nrow(X_vec_ROT),,,drop=F]-X_vec_ROT[1:(nrow(X_vec_ROT)-1),,,drop=F]
      
      sume_dxx=0
      sume_xx=0
      for(i in 2:N_max){
        sume_xx<-sume_xx+ matrix(X_vec_ROT[i-1,,],nrow=dim) %*% t(matrix(X_vec_ROT[i-1,,],nrow=dim))
        if(i>1) sume_dxx<-sume_dxx+ matrix(dX_vec_ROT[i-1,,],nrow = dim) %*% t(matrix(X_vec_ROT[i-1,,],nrow=dim))
      }
      A_est<- (1/dt)* sume_dxx %*% solve(sume_xx)
      
      X_vec_ROT_mod<-array(NA, dim=dim(X_vec_ROT[2:nrow(X_vec_ROT),,,drop=F]))
      for(i in 1:(N_max-1)){
        X_vec_ROT_mod[i,,]<-(A_est*dt+diag(dim)) %*% X_vec_ROT[i,,]
      }
      
      dX_vec_ROT=X_vec_ROT[2:nrow(X_vec_ROT),,,drop=F]- X_vec_ROT_mod
      cur_G=0
      for(t in 1:n_sim_traj){
        cur_G= cur_G+ t(matrix(dX_vec_ROT[,,t],ncol=dim))%*% matrix(dX_vec_ROT[,,t],ncol = dim)/(Tmax-dt)
        #print(t(matrix(dX_vec_ROT[,,t],ncol=dim))%*% matrix(dX_vec_ROT[,,t],ncol = dim)/(Tmax-dt))
      }
      GGT_est<- cur_G/n_sim_traj
      
      multi_sinkho_G[r,it,s]<-mean(abs(GGT_est- G%*%t(G)))/mean(abs( G%*%t(G)))
      multi_sinkho_A[r,it,s]<-mean(abs(A_est- A))/mean(abs(A))
      print(it)
      if(it %% 5 ==0){
        print(GGT_est)
        print(A_est)
      }
    }
    print(paste("REP NUMBER:", r, "IS DONE!!!!!"))
  }
}

multi_sinkho_A<-multi_sinkho_A*100
multi_sinkho_G<-multi_sinkho_G*100

plot(colmeans(multi_sinkho_A[,,1]),type="b",ylim=c(0,max(multi_sinkho_A,multi_sinkho_G)),ylab="MAPE",xlab="iteration")
lines(colmeans(multi_sinkho_G[,,1]),type="b",col="red",pch=19)
arrows(1:iter, colMins(multi_sinkho_A[,,1],value = T), 1:iter, colMaxs(multi_sinkho_A[,,1],value = T), length=0.05, angle=90, code=3)
arrows(1:iter, colMins(multi_sinkho_G[,,1],value = T), 1:iter, colMaxs(multi_sinkho_G[,,1],value = T), length=0.05, angle=90, code=3,col="red")
legend("topright", legend=c("A1", "G1"), col=c("black","red"), cex=0.8,pch=c(1,19))


# compute error at sample points

drift_magnitude_quadratic <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/drift_magnitude_quadratic.csv")
cor_dat <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/metric=grid_cos_p0-gmm_pointwise.csv")
mae_dat <- read.csv("C:/Users/joeja/Desktop/researchMasters/ParamEstMarginals/AISTATS_2026/metric=grid_mae_p0-gmm_pointwise.csv")

mae_dat<-mae_dat[mae_dat$potential=="poly" & mae_dat$method=="appex" & mae_dat$seed==1001,]
cor_dat<-cor_dat[cor_dat$potential=="poly" & cor_dat$method=="appex" & cor_dat$seed==1001,]

drift_magnitude_quadratic$x<-round(drift_magnitude_quadratic$x,2)
drift_magnitude_quadratic$y<-round(drift_magnitude_quadratic$y,2)

cor_dat$x<-round(cor_dat$x,2)
cor_dat$y<-round(cor_dat$y,2)

mae_dat$x<-round(mae_dat$x,2)
mae_dat$y<-round(mae_dat$y,2)


A_gt=matrix(c(-10,0,0,-10),nrow=2)
H_gt=matrix(c(0.2,0,0,0.2),nrow=2)

A_est=matrix(c(-10.01181,-0.06658,-0.0268,-10.003945),nrow=2)
H_est=matrix(c(0.283152,-0.0064436,-0.0064436,0.28571),nrow=2)

mae_dat$drift_MAE<-NA
cor_dat$drift_cos<-NA

for(i in 1:nrow(mae_dat)){
  xcur<-matrix(c(mae_dat$x[i],mae_dat$y[i]),ncol=1)
  drift_est<- A_est %*% xcur
  drift_gt<-A_gt %*% xcur
  
  mae_dat$drift_MAE[i]<-sum(abs(drift_est- drift_gt))/drift_magnitude_quadratic$magnitude[drift_magnitude_quadratic$x==xcur[1] & drift_magnitude_quadratic$y==xcur[2]]
}


for(i in 1:nrow(cor_dat)){
  xcur<-matrix(c(cor_dat$x[i],cor_dat$y[i]),ncol=1)
  drift_est<- A_est %*% xcur
  drift_gt<-A_gt %*% xcur
  
  cor_dat$drift_cos[i]<-sum(t(drift_est) %*% drift_gt)/(sqrt(t(drift_gt) %*% drift_gt) * sqrt(t(drift_est) %*% drift_est))
}


