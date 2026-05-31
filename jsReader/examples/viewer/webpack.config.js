const webpack = require('webpack')
const path = require('path')

const config = {
  /*devtool: 'inline-source-map',
  mode:'development',*/
  mode: 'production',
  entry: path.resolve(__dirname, './index.js'),
  output: {
    path: path.resolve(__dirname, './public'),
    filename: './index.js'
  },
  resolve: {
    fallback: {
      zlib: require.resolve('browserify-zlib'),
      stream: require.resolve('stream-browserify'),
      buffer: require.resolve('buffer/'),
      events: require.resolve('events/'),
      assert: require.resolve('assert/'),
      path: require.resolve('path-browserify'),
      canvas: false
    }
  },
  plugins: [
    // fix "process is not defined" error:
    new webpack.ProvidePlugin({
      process: 'process/browser',
      Buffer: ['buffer', 'Buffer']
    }),
    new webpack.NormalModuleReplacementPlugin(
      /prismarine-viewer\/viewer\/lib\/utils(\.js)?$/,
      (resource) => {
        resource.request = resource.request.replace(
          /utils(\.js)?$/,
          'utils.web.js'
        )
      }
    ),
    // Assets already in public/ from previous build — CopyPlugin removed to avoid EMFILE
  ],

  devServer: {
    contentBase: path.resolve(__dirname, './public'),
    compress: true,
    inline: true,
    // open: true,
    hot: true,
    watchOptions: {
      ignored: /node_modules/
    }
  }
}

module.exports = config
