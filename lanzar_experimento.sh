#!/bin/bash
URL="https://discord.com/api/webhooks/1511011308869390510/t_tVqF6XjLJg1Yo3vwKUTgH9P5RI9NwQEKFIWYxFF4eHePOn_2ojh_Dq89_0ckA_Ngy7"

# 1. Alerta de Inicio
curl -H "Content-Type: application/json" -X POST -d '{"content": "🚀 **MethaneSAT:** Iniciando `run_experiment.py` con Cross-Attention..."}' $URL

# 2. Correr el modelo
python run_experiment.py --config config/msat_cs_multiscale.yaml

# 3. Alerta de Fin (Monitoreando el estado de salida)
if [ $? -eq 0 ]; then
  curl -H "Content-Type: application/json" -X POST -d '{"content": "✅ **¡Éxito!** El entrenamiento del modelo terminó correctamente. Todos los folds completados."}' $URL
else
  curl -H "Content-Type: application/json" -X POST -d '{"content": "❌ **¡ALERTA!** El script se cayó con un error. Revisa el archivo `experiment.log` de inmediato."}' $URL
fi