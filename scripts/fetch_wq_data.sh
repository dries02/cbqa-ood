splits=(train dev)                      # test is loaded elsewhere due to annotations

outdir=data
mkdir -p $outdir 
mkdir -p $outdir/"webquestions"         # other datasets are loaded elsewhere

tmpdir=tmp
mkdir -p $tmpdir                        # don't crash if exists

for split in ${splits[@]}; do
  name="webquestions-${split}.json"
  url=https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-${name}.gz

  wget -q --show-progress -O $tmpdir/$name.gz $url
  gunzip -f $tmpdir/$name.gz            # unzip and delete .gz
                                       
  infile="$tmpdir/$name"
  outfile="$outdir/webquestions/$name"

  jq '[ .[] | { question, answers } ]' $infile > $outfile     # remove unnecessary columns relating to Karpukhin et al.

  echo handled $outfile
done

rm -rf $tmpdir

# References:
# https://arxiv.org/pdf/2004.04906 (Karpukhin et al. 2020)
# https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py
