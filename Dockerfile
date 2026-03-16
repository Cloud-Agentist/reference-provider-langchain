# Reference Provider: LangChain.js
#
# Build from reference-provider-langchain/ directory:
#   docker build -t reference-provider-langchain .
#
# Run:
#   docker run -p 8081:8081 \
#     -e OPENAI_API_KEY=sk-... \
#     reference-provider-langchain

FROM node:22-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts
COPY tsconfig.json ./
COPY src/ ./src/
RUN npm run build

FROM node:22-alpine AS runtime
WORKDIR /app
ENV PORT=8081 NODE_ENV=production

RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist/
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules/
COPY --from=builder --chown=nodejs:nodejs /app/package.json ./
USER nodejs
EXPOSE 8081
CMD ["node", "dist/index.js"]
