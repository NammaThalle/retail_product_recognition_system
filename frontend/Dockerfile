# Frontend Dockerfile
FROM node:18 as build

# Set working directory in the container
WORKDIR /app

# Copy the package.json and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy the rest of the app source code
COPY . .

# Build the React app for production
RUN npm run build

# Serve the app using a minimal web server like nginx
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html

# Expose the port on which the frontend runs
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
