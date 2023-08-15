
from torchmetrics import CharErrorRate
from extract import *
from symmetry import *
from patches import  *
from metrics import *
charm=CharErrorRate()
transform_tens = transforms.ToTensor()
def read_image(path):
    img= cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def compare_metric(preds,label):
    maxx=0
    res=""
    for strs in preds:
        score=TED_similarity_score(strs,label)
        if score>=maxx:
            maxx=score
            res=strs
    return maxx,res

def evaluate_without_siamese(df,img_path,mask_path,model):
    """
    input args: takes dataframe as the input
    returns accuracy,character error rate(CER)
    """
    ted=0
    cer=0
    correct=0
    for i in range(len(df)):
        img_pth=os.path.join(img_path,df["file"][i])
        mask_pth=os.path.join(mask_path,df["masks"][i])

        img=read_image(img_pth)
        res=extract_license_UFPR_plate_number(mask_pth)
        #using spts model
        preds,recog=model.run_on_image(img)
        
        #pred_sample_don.append(gen_tex)
        p,gen_tex=compare_metric(recog,res)
        #print(recog,res,p)
        ted+=p
        cer+=charm([res],[gen_tex])
    #print("percetage of correct is:",correct/len(df))
    return ted/len(df),cer/len(df)
#Using Symmetry on tExt Spotting Model

def evaluate_with_siames(df,img_path,mask_path,model1,model2):
    """
    input args: takes dataframe as the input
    returns accuracy,character error rate(CER)
    model1 is inference
    model2 is for symmetry
    """
    ted=0 #tree-distance to calculate the accuracy
    cer=0
    m=0
    f=0
    model2=model2.cuda()
    for i in range(len(df)):
        img_pth=os.path.join(img_path,df["file"][i])
        mask_pth=os.path.join(mask_path,df["masks"][i])
        res=extract_license_UFPR_plate_number(mask_pth)
        patches=Datapatches(img_pth)
        patches2=data_triple_loss_pair(patches)
        #print("image :",i,res[0])
        #print(res)
        charc=""
        for anc,pst,neg in patches2:
            anc,pst,neg=cv2.resize(anc, (224, 224)),cv2.resize(pst, (224, 224)),cv2.resize(neg, (224, 224))
            anc1=Image.fromarray(anc)
            pst1=Image.fromarray(pst)
            neg1=Image.fromarray(neg)
            pred_a,pred_b=test(transform_tens(anc1),transform_tens(pst1),transform_tens(neg1),0,model1,model2)
            x,a=compare_metric(pred_a,res)
            y,b=compare_metric(pred_b,res)
            #print(pred_a,pred_b)
            #print(a,b)
            if x>=y:
                if x>=m:
                    charc=a
                    m=x
            elif y>x:
                if y>=m:
                    charc=b
                    m=y

        #print(m)
        #print(charc,res,m)
        ted=ted+m
        if res and charc:
        #print(charc,res)
            cer=cer+charm([res],[charc])



    #print("percentage of correct detection is:",f)
    #print(cer)
    return ted/len(df),cer/len(df)
