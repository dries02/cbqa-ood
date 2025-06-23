# datasets=(webquestions trivia nq)
splits=(train dev)                     # test is loaded elsewhere due to annotations

datasets=(webquestions)                # uncomment for debugging
# splits=(dev)                         # uncomment for debugging

outdir=data
mkdir -p $outdir 

tmpdir=tmp
mkdir -p $tmpdir                       # don't crash if exists


for dataset in ${datasets[@]}; do
  mkdir -p $outdir/$dataset

  for split in ${splits[@]}; do
    name="${dataset}-${split}.json"
    url=https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-${name}.gz

    wget -q --show-progress -O $tmpdir/$name.gz $url
    gunzip -f $tmpdir/$name.gz         # unzip and delete .gz
                                       

    infile="$tmpdir/$name"
    outfile="$outdir/$dataset/$name"

    jq '[ .[] | { question, answers } ]' $infile > $outfile     # remove unnecessary columns relating to Karpukhin et al.
                                                                # TODO fix this for larger files
                                                                # some issues with streams...

    # stackoverflow.com/questions/39232060/process-large-json-stream-with-jq

    echo handled $outfile
  done
done

rm -rf $tmpdir

# References:
# https://arxiv.org/pdf/2004.04906 (Karpukhin et al. 2020)
# https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py
