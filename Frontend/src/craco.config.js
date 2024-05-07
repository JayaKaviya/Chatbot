const path = require('path');
module.exports = {
    webpack: {
      configure: {
        resolve: {
          fallback: { "buffer": require.resolve("buffer/") }
        }
      }
    }
  };
  