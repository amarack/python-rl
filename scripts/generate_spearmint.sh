
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ && pwd )"
OUTDIR=$1
EXPFILE=$2

AGENT_NAME=`cat $EXPFILE | tr "\n" " " | sed -e 's/{.*"agent"[ ]*:.*"name"[ ]*:[ ]*"\([^"]*\)".*$/\1/g'`
mkdir $OUTDIR
mkdir ${OUTDIR}/${AGENT_NAME}

echo $DIR
cat ${DIR}/scripts/spearmint_template.py | sed -e s:'pyrl_path = "###"':'pyrl_path = "'${DIR}'"':g > ${OUTDIR}/${AGENT_NAME}/${AGENT_NAME}.py
cp $EXPFILE ${OUTDIR}/${AGENT_NAME}/experiment.json

python spearmint_config.py ${AGENT_NAME} $EXPFILE > ${OUTDIR}/${AGENT_NAME}/config.pb


