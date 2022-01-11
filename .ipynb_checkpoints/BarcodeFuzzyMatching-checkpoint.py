#usage: python BarcodeFuzzyMatching.py index_sample_key.txt multiseq_index_seqs.txt fastq_R1_file.fq cell_barcodes.txt out_dir cell_barcode_len index_len flipped_indicator 
#assumes your fastqs are catt'd, reads are paired, gunzipped, and named with R1 and R2
#cell_barcode_len is the number of bases from read start in cell barcode for 10X, this is 16
#index_len is the length of the R2 index, for multiseq this is 8
#depends on cell barcodes and index barcodes being the first part of the reads
#this can be altered using longer cell_barcode_len or index_len
#Also could adjust mismatch tolerance by altering min_df in script
#flipped_indicator == 1 if you used the multiseq index plate in the wrong orientation (oops)
#else 0 or leave it empty

#usage example: python BarcodeFuzzyMatching.py /path/to/this/repo/MultiseqSamplesExample.txt /path/to/this/repo/MultiseqIndices.txt /path/to/sampleMULTIseq_R1.fastq  /path/to/cellranger/outs/filtered_feature_bc_matrix/barcodes.tsv.gz /path/to/output/dir/ 16 8 0

import re
import fuzzywuzzy
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import sys
import os
import tqdm
import numpy as np
import sparse_dot_topn
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from scipy import io
from scipy.sparse import csr_matrix

#Path to MultiseqIndices.txt or your index file
keyfile=sys.argv[1]
key=pd.read_csv(keyfile,sep='\t')
key=key.replace(np.nan, '', regex=True)
key.index=key['Multiseq_Index']

#indexfile='/wynton/group/ye/mtschmitz/macaquedevbrain/MULTISEQmacaque/MultiseqIndices.txt'
indexfile=sys.argv[2]

#assumes your fastqs are catt'd, reads are paired, gunzipped, and named with R1 and R2
fq1=sys.argv[3]
fq2=re.sub('_R1_','_R2_',fq1)

#cellfile='/path/to/cellranger/outs/filtered_feature_bc_matrix/barcodes.tsv.gz'
cellfile=sys.argv[4]

#outfile='/path/to/out/dir/'
outfile=sys.argv[5]

#length of forward read to use
f_len=sys.argv[6] if len(sys.argv) > 6 else 16
#length of reverse read to use
r_len=sys.argv[7] if len(sys.argv) > 6 else 8

#Just in case you flipped your index plate when you multichanneled indices
pf=sys.argv[8] if len(sys.argv) > 8 else '0'
plate_flipped=True if pf == '1' else False
Indices=pd.read_csv(indexfile,sep='\t')
Indices.index=Indices.index+1
if plate_flipped:
    eighttwelve=pd.read_csv('./8-12IndexConversion.txt',sep='\t')
    etdict=dict(zip(eighttwelve['8-Index'],eighttwelve['12-Index']))
    key['Multiseq_Index']=[etdict[x] for x in key['Multiseq_Index']]
#Somewhat adhoc for my multiseq format key
keydict=dict(zip(key['Multiseq_Index'],key['Sample']))

#Read in Fastq information
#Adapted from: https://www.biostars.org/p/317524/
#Function to parse fastq
def processfq(lines=None):
    ks = ['name', 'sequence', 'optional', 'quality']
    return {k: v for k, v in zip(ks, lines)}

n = 4
readpairs=[]
i=0
with open(fq1, 'r') as fh:
    with open(fq2, 'r') as rh:
        flines = []
        rlines = []
        linef = fh.readline()
        liner = rh.readline()
        while linef and liner:
            flines.append(linef.rstrip())
            rlines.append(liner.rstrip())
            if (len(flines) == n) and (len(rlines) == n):
                recordf = processfq(flines)
                recordr = processfq(rlines)
                readpairs.append([recordf['sequence'][0:f_len],recordr['sequence'][0:r_len]])
                flines = []
                rlines = []
            i+=1
            if i%100000==0:
                print(i)
            linef = fh.readline()
            liner = rh.readline()

df=pd.DataFrame(readpairs)
"""cells=[]
with open(cellfile) as f:
    for l in f.readlines():
        if len(l)>2:
            cells.append(re.sub('\n','',l))"""
cells=list(pd.read_csv(cellfile, header=None, index_col=False, sep='\t')[0])

cellset=list(set(cells))
Indices=Indices.loc[Indices.index.isin(key.index),:]
inds=list(Indices['Barcode_Sequence'])
inddictrev=dict(zip(Indices.index,Indices['Barcode_Sequence']))
inddict=dict(zip(inddictrev.values(),inddictrev.keys()))

celldictrev=dict(enumerate(cellset))
celldict=dict(zip(celldictrev.values(),celldictrev.keys()))

mat=np.zeros((len(cellset),len(inds)),dtype=np.int32)

bcfixer=[process.extractOne(x,inddictrev) for x in tqdm.tqdm(df[1].unique())]
bcfixerdict=dict(zip(df[1].unique(),bcfixer))
bcfixed=[bcfixerdict[x] for x in df[1]]

df=df.loc[[x is not None for x in bcfixed],:]
df[1]=[x[2] for x in bcfixed if x is not None]
print(df,flush=True)

def ngrams(string, n=8):
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(cellset+list(df[0]))

tf_idf_matrix_ref=tf_idf_matrix[:len(cellset),:]

tf_idf_matrix_query=tf_idf_matrix[len(cellset):,:]
#print(cellset,flush=True)
#print(tf_idf_matrix_ref,flush=True)
#print('query:',flush=True)
#print(tf_idf_matrix_query,flush=True)
#print(tf_idf_matrix_ref.shape,tf_idf_matrix_query.shape)
def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

def get_matches_df(sparse_matrix, name_vectorx,name_vectory, top=100):
    non_zeros = sparse_matrix.nonzero()
    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    sparserows=sparse_matrix.argmax(1).A1
    sparsecols=list(range(len(name_vectory)))
    left_side = name_vectorx
    right_side = np.array(name_vectory)[sparserows]
    similarity=[]
    for x,y in zip(list(range(len(name_vectorx))),sparserows):
        similarity.append(sparse_matrix[x,y])
    
    return pd.DataFrame({'original': left_side,
                          'matched': right_side,
                           'similarity': similarity})

matches = awesome_cossim_top(tf_idf_matrix_query,tf_idf_matrix_ref.transpose(), 1)

matchdf=get_matches_df(matches,list(df[0]),cellset,False)

df[0]=matchdf['matched']

df=df.loc[list(matchdf['similarity']>.7),:]

mat=df.groupby([0,1]).size().unstack(fill_value=0)
mat.columns=[keydict[x] for x in mat.columns]
mat.astype('int')
mat.to_csv(os.path.join(outfile,'MULTIseq_counts.txt'),sep='\t',header=True,index=True)

if not os.path.exists(os.path.join(outfile,'multiseq_outs')):
    os.mkdir(os.path.join(outfile,'multiseq_outs'))

scipy.io.mmwrite(os.path.join(outfile,'multiseq_outs','matrix.mtx'),scipy.sparse.csr_matrix(np.array(mat).T))
with open(os.path.join(outfile,'multiseq_outs','barcodes.tsv'),'w') as f:
    for ind,i in enumerate(mat.index):
        f.write(i)
        f.write("\n")
        
with open(os.path.join(outfile,'multiseq_outs','features.tsv'),'w') as f:
    for ind,i in enumerate(mat.columns):
        f.write(str(ind)+'\t'+str(i)+'\tAntibody Capture')
        f.write("\n")