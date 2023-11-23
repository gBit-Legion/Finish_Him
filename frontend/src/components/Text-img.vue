<template>
  <div class="map" id="box">
    <h1>{{ title }}</h1>
    <img class="img" :src="require('../assets/' + imgURL)" width="700" />
  </div>
</template>

<style scoped>
@import url('https://fonts.googleapis.com/css?family=Raleway:200');

.img {
    border-radius: 20px;
}

.map {
  width: 40%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  --borderWidth: 10px;
  background: white;
  position: relative;
  border-radius: var(--borderWidth);
}

.map:after {
  content: '';
  position: absolute;
  top: calc(-1 * var(--borderWidth));
  left: calc(-1 * var(--borderWidth));
  height: calc(100% + var(--borderWidth) * 2);
  width: calc(100% + var(--borderWidth) * 2);
  background: linear-gradient(60deg, #f79533, #f37055, #ef4e7b, #a166ab, #5073b8, #1098ad, #07b39b, #6fba82);
  border-radius: calc(2 * var(--borderWidth));
  z-index: -1;
  animation: animatedgradient 3s ease alternate infinite;
  background-size: 300% 300%;
}

#box {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40%;
  height: 600px;
  color: #000000;
  font-family: 'Raleway';
  font-size: 1rem;
}

@keyframes animatedgradient {
	0% {
		background-position: 0% 50%;
	}
	50% {
		background-position: 100% 50%;
	}
	100% {
		background-position: 0% 50%;
	}
}


</style>

<script>
export default {
  props: ["imgURL", "title"],
};
</script>
