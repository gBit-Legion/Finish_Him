<template>
  <nav>
    <router-link to="/">Drop</router-link> |
    <router-link to="/props">props</router-link>
  </nav>
  <router-view/>
</template>

<style>
body {
  margin: 0 auto;
  background: black;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #5c5e80;

}

nav {
  padding: 30px;
}

nav a {
  font-weight: bold;
  color: #5c5e80;
}

nav a.router-link-exact-active {
  color: #42b983;
}
</style>
