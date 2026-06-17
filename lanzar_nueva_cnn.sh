#!/bin/bash
URL="https://discord.com/api/webhooks/1511011308869390510/t_tVqF6XjLJg1Yo3vwKUTgH9P5RI9NwQEKFIWYxFF4eHePOn_2ojh_Dq89_0ckA_Ngy7"

# 1. Alerta de Inicio
curl -s -H "Content-Type: application/json" -X POST -d '{"content": "🚀 **MethaneSAT:** Iniciando `run_experiment.py` con nueva combinedCNN..."}' "$URL" > /dev/null

# 2. Correr el modelo capturando la salida en tiempo real
# Usamos tee para guardarlo en log, mientras que en el bucle se imprime a tmux y se analiza
python run_experiment.py --config config/msat_cs_multiscale.yaml 2>&1 | tee experiment.log | while IFS= read -r line; do
  # Imprimir a la pantalla (tmux)
  echo "$line"
  
  # Si detecta inicio de fold, manda alerta
  if [[ "$line" == *"Starting experiment with command:"* ]]; then
      FOLD=$(echo "$line" | grep -o "\-\-fold=[0-9]*" | cut -d'=' -f2)
      curl -s -H "Content-Type: application/json" -X POST -d "{\"content\": \"🔄 **Progreso:** Iniciando entrenamiento del Fold $FOLD...\"}" "$URL" > /dev/null
  fi
done

# Capturar el exit code original de Python
EXIT_CODE=${PIPESTATUS[0]}

# 3. Alerta de Fin (Monitoreando el estado de salida)
if [ $EXIT_CODE -eq 0 ]; then
  # Extraer la línea de resultados del mejor experimento
  BEST_EXPERIMENT=$(grep "Best experiment:" experiment.log | tail -n 1)
  
  if [ -n "$BEST_EXPERIMENT" ]; then
    # Extraer métricas usando grep
    F1=$(echo "$BEST_EXPERIMENT" | grep -o "'f1': [0-9.]*" | cut -d' ' -f2)
    ACC=$(echo "$BEST_EXPERIMENT" | grep -o "'acc': [0-9.]*" | cut -d' ' -f2)
    
    curl -s -H "Content-Type: application/json" -X POST -d "{\"content\": \"✅ **¡Éxito!** Todos los folds completados.\n📊 **Métricas del Mejor Fold:**\n- **F1 Score**: $F1\n- **Accuracy**: $ACC\"}" "$URL" > /dev/null
  else
    curl -s -H "Content-Type: application/json" -X POST -d '{"content": "✅ **¡Éxito!** El entrenamiento del modelo terminó correctamente. Todos los folds completados."}' "$URL" > /dev/null
  fi
else
  # Extraer últimas líneas de error, escapar comillas y reemplazar saltos de línea por \n textual
  ERROR_ESCAPED=$(tail -n 15 experiment.log | sed 's/"/\\"/g' | awk '{printf "%s\\n", $0}')
  
  curl -s -H "Content-Type: application/json" -X POST -d "{\"content\": \"❌ **¡ALERTA!** El script falló.\n**Últimas líneas del error en tmux:**\n\`\`\`\n${ERROR_ESCAPED}\n\`\`\`\"}" "$URL" > /dev/null
fi