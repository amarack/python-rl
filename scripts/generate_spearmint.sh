#####################################################################
#
# Generate Spearmint configuration/experiment 
# to optimize parameters for some pyRL experiment file.
#
# Run with:
# sh scripts/generate_spearmint.sh /path/to/output experimentfile.json
#
# This will create all the necessary files in /path/to/output 
# spearmint needs to run the optimization. It will use experimentfile.json 
# as the template for the experiment.
#
# Note: Make sure *at least* one parameter for the algorithm 
# is *NOT* specified in the experiment json file. This code 
# will use whatever values are specified in the json file and 
# will *ONLY* optimize the parameters that are not given in the json.
#
# Then, to run the experiment go to spearmint directory and run:
# python spearmint_sync.py --method=GPEIOptChooser --method-args=noiseless=1 /path/to/output/
#####################################################################

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ && pwd )"
OUTDIR=$1
EXPFILE=$2

AGENT_NAME=`cat $EXPFILE | tr "\n" " " | sed -e 's/{.*"agent"[ ]*:.*"name"[ ]*:[ ]*"\([^"]*\)".*$/\1/g'`
mkdir $OUTDIR

cat ${DIR}/scripts/spearmint_template.py | sed -e s:'pyrl_path = "###"':'pyrl_path = "'${DIR}'"':g > ${OUTDIR}/"${AGENT_NAME}.py"
cp $EXPFILE ${OUTDIR}/experiment.json

python ${DIR}/scripts/spearmint_config.py "${AGENT_NAME}" $EXPFILE > ${OUTDIR}/config.pb


