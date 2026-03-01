FROM node:20-alpine AS builder
WORKDIR /app
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci
COPY web/frontend/ .
RUN npm run build

FROM nginx:alpine
RUN apk add --no-cache curl
COPY web/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist/ /usr/share/nginx/html/
RUN curl -L -o /usr/share/nginx/html/fastconformer_ar_ctc_q8.onnx \
    https://github.com/yazinsai/offline-tarteel/releases/download/v0.1.0/fastconformer_ar_ctc_q8.onnx
EXPOSE 5000
