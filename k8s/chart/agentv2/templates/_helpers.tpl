{{- define "agentv2.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agentv2.labels" -}}
app.kubernetes.io/name: {{ include "agentv2.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "agentv2.selectorLabels" -}}
app.kubernetes.io/name: {{ include "agentv2.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "agentv2.redisName" -}}
{{ include "agentv2.name" . }}-redis
{{- end -}}
