const halfButton = document.getElementsByTagName("button")[0];
const fullButton = document.getElementsByTagName("button")[1];
const form = document.getElementsByTagName("form")[0];
// const colorBar=document.getElementsByClassName("color-bar")[0];
const four=document.getElementsByClassName("4th")[0];
const five=document.getElementsByClassName("5th")[0];
const six=document.getElementsByClassName("6th")[0];
const zero=document.getElementsByClassName("zero")[0];

const isPrimary = document.getElementsByClassName("isPrimary")[0];

fullButton.onclick = function () {
    fullButton.style.border = " 0.5px solid rgb(83, 78, 78)";
    fullButton.style.backgroundColor = "#179aff";
    fullButton.style.color = "white";
    halfButton.style.backgroundColor = "white";
    halfButton.style.color = "rgb(83, 78, 78)";
    form.style.display = "block";
    four.style.display="flex";

//    document.getElementById("bathtubs").setAttribute('value','1');
//    document.getElementById("showers").setAttribute('value','1');

    five.style.display="flex";
    six.style.display="flex";
    zero.style.display="none";
document.getElementById("isPrimary").style.display="block";
    //   document.getElementsByTagName("fieldset")[0].style.width="fit-content";
    document.getElementsByTagName("fieldset")[0].style.width="700px";

    document.getElementById("bathtubs").removeAttribute("disabled");
    document.getElementById("showers").removeAttribute("disabled");
    document.getElementById("tubs").removeAttribute("disabled");


}
halfButton.onclick = function () {
    halfButton.style.backgroundColor = "#179aff";
    halfButton.style.color = "white";
    halfButton.style.border = " 0.5px solid rgb(83, 78, 78)";
    fullButton.style.backgroundColor = "white";
    fullButton.style.color = "rgb(83, 78, 78)";
    form.style.display = "block";

    document.getElementById("bathtubs").disabled=true;
    document.getElementById("showers").disabled=true;
    document.getElementById("tubs").disabled=true;



//   document.getElementById("bathtubs").setAttribute('value','0');
//   document.getElementById("showers").setAttribute('value','0');

    four.style.display="none";
    five.style.display="none";
    six.style.display="none";
    isPrimary.style.display="flex";
    zero.style.display="flex";
    isPrimary.style.display="none";

    document.getElementsByTagName("fieldset")[0].style.width="700px";

 
}
function showIsPrimary(isPrimary) {
   document.getElementById(isPrimary).style.display = "none";
}


// more effcitent way of doing this?? doing this way can get messy.


subArr=document.querySelectorAll(".sub");
addArr=document.querySelectorAll(".add");
valueArr=document.querySelectorAll(".quantity");
quantity=[];
for(let i=0;i<valueArr.length;i++){
    quantity.push(0);
}
// make it global/reachale by all, having it local would countuine to reset it back to this value


for(let i=0;i<subArr.length;i++){
    subArr[i].onmousedown =function(){
        subArr[i].style.color = "black";
    }
    subArr[i].onmouseup =function(){
        subArr[i].style.color = "lightblue";
    }
    subArr[i].onclick=function (){

        if(quantity[i]>0){
           quantity[i]--;
        }
        valueArr[i].value=quantity[i];

   }
}
for(let i=0;i<addArr.length;i++){
    addArr[i].onmousedown =function(){
        addArr[i].style.color = "black";
    }
    addArr[i].onmouseup =function(){
        addArr[i].style.color = "lightblue";
    }
addArr[i].onclick=function (){
    quantity[i]++;
    valueArr[i].value=quantity[i];

}
}