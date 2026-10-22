"use strict";
const $ = id => document.getElementById(id);
const ACTIVE = new Set(["starting", "running", "stopping"]);
const statusNames = {starting:"正在准备",running:"训练中",stopping:"正在保存",stopped:"已停止",completed:"已完成",failed:"训练失败",interrupted:"服务曾中断"};
let boot = null, state = {active_id:null,run:null,runs:[],data_job:{status:"idle"}}, selectedId = null, online = false, pending = false, closed = false, toastTimer = null, lastLog = "";
const number = n => Number(n).toLocaleString("zh-CN");
const loss = n => Number.isFinite(n) ? n.toFixed(4) : "—";
function duration(n) {
  if (!Number.isFinite(n)) return "—";
  n = Math.max(0,Math.round(n));
  if (n < 60) return `${n} 秒`;
  if (n < 3600) return `${Math.floor(n/60)} 分 ${n%60} 秒`;
  return `${Math.floor(n/3600)} 小时 ${Math.floor(n%3600/60)} 分`;
}
function notify(message, type="info") {
  clearTimeout(toastTimer); $("toast-message").textContent=message; $("toast").dataset.type=type; $("toast").hidden=false;
  if(type!=="error") toastTimer=setTimeout(()=>$("toast").hidden=true,5000);
}
async function request(path, body) {
  const options = body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json","X-SmallAI-Token":boot.csrf_token},body:JSON.stringify(body)};
  const response=await fetch(path,options);
  const value=await response.json();
  if(!response.ok) throw new Error(value.error || `请求失败 (${response.status})`);
  return value;
}
function setOnline(value) {
  online=value; $("connection").dataset.online=String(value);
  $("connection-text").textContent=value?"已连接本机":closed?"服务已关闭":"连接已断开";
}
function addOption(select,value,label) {
  const option=document.createElement("option"); option.value=value; option.textContent=label; select.append(option);
}
async function bootstrap(first=false) {
  const previousModel=$("model").value;
  boot=await request("/api/bootstrap");
  $("model").replaceChildren();
  for(const model of boot.models) addOption($("model"),model.id,model.title);
  if(boot.models.some(m=>m.id===previousModel)) $("model").value=previousModel;
  if(first) state.data_job=boot.data_job||{status:"idle"};
  if(first) {$("threads").value=boot.default_threads;$("threads").max=boot.max_threads;}
  stageChanged(false);
}
function stageChanged(resetRate=true) {
  if(!boot) return;
  const sft=$("stage").value==="sft", oldDataset=$("dataset").value;
  $("model-field").hidden=sft; $("checkpoint-field").hidden=!sft; $("block-field").hidden=sft; $("sft-block-hint").hidden=!sft;
  $("stage-hint").textContent=sft?"在自己的权重上学习对话与工具调用。":"从随机权重开始，学习文本规律。";
  if(resetRate) $("learning-rate").value=sft?"0.0001":"0.0003";
  $("dataset").replaceChildren();
  for(const dataset of boot.datasets.filter(d=>d.stage===$("stage").value)) {
    let title=dataset.display_name || (dataset.path===`data/prepared/${dataset.stage}`?(sft?"示例对话与工具调用":"示例中文语料"):dataset.path);
    addOption($("dataset"),dataset.path,title);
  }
  if([...$("dataset").options].some(o=>o.value===oldDataset)) $("dataset").value=oldDataset;
  updateHints(); renderControls();
}
function updateHints() {
  if(!boot) return;
  const data=boot.datasets.find(d=>d.path===$("dataset").value);
  const model=boot.models.find(m=>m.id===$("model").value);
  const params=model?(model.non_embedding_parameters+model.hidden_size*(data?.vocab_size||16384)):0;
  $("model-hint").textContent=model?`${(params/10000).toLocaleString("zh-CN",{maximumFractionDigits:1})} 万参数 · ${model.layers} 层` : "";
  const previous=$("checkpoint").value;
  const compatible=boot.checkpoints.filter(c=>c.tokenizer_sha256===data?.tokenizer_sha256);
  $("checkpoint").replaceChildren();
  for(const cp of compatible)addOption($("checkpoint"),cp.path,`${cp.label}${cp.step==null?"":` · ${cp.step} 步`}`);
  if(compatible.some(c=>c.path===previous))$("checkpoint").value=previous;
  else if(compatible.some(c=>c.path===boot.default_checkpoint))$("checkpoint").value=boot.default_checkpoint;
  const cp=compatible.find(c=>c.path===$("checkpoint").value);
  $("checkpoint-hint").textContent=cp?`${(cp.parameters/10000).toLocaleString("zh-CN",{maximumFractionDigits:1})} 万参数 · 分词器匹配`:"请先用这份公开数据的分词器完成一次从零预训练。";
  $("dataset-hint").textContent=data?`训练 ${number(data.splits.train.documents)} 条 · 验证 ${number(data.splits.val.documents)} 条` : "没有可用数据，请按 README 准备数据。";
  if(data?.stage==="sft") $("sft-block-hint").textContent=`对话块长度：${data.block_size} token，由数据集决定。`;
  $("dataset-sources").replaceChildren();
  for(const source of data?.sources||[]) {
    try {const url=new URL(source.url);if(url.protocol!=="https:"||url.hostname!=="huggingface.co")continue;const a=document.createElement("a");a.href=url.href;a.target="_blank";a.rel="noopener noreferrer";a.textContent=`${source.dataset.split("/")[1]} · ${source.license}`;$("dataset-sources").append(a);}catch{}
  }
  const targetTokens=(Number($("steps").value)||0)*(Number($("batch-size").value)||1)*(Number($("accumulation").value)||1)*(Number($("block-size").value)||128);
  $("step-hint").textContent=data?.stage==="pretrain"?`训练集约 ${number(data.splits.train.tokens)} token；当前计划约 ${number(targetTokens)} token。`:"一次参数更新计为一步；按对话随机采样。";
  renderControls();
}
function renderControls() {
  const trainingBusy=Boolean(state.active_id), dataBusy=["running","cancelling"].includes(state.data_job?.status);
  const busy=trainingBusy||dataBusy, run=state.run;
  $("settings-fields").disabled=busy||pending||closed;
  const selectedData=boot?.datasets.find(d=>d.path===$("dataset").value);
  const ready=selectedData?.tokenizer_ready && $("dataset").value && ($("stage").value!=="sft" || $("checkpoint").value);
  $("start-button").disabled=!online||busy||pending||!ready;
  $("start-label").textContent=dataBusy?"正在准备数据":trainingBusy?"训练正在进行":pending?"正在启动…":"开始训练";
  $("form-note").textContent=busy?"停止并保存后，可继续训练或新建实验。":"每次新训练都会创建独立的输出目录。";
  $("stop-button").hidden=!run||run.id!==state.active_id;
  $("stop-button").disabled=!online||pending||run?.status==="stopping";
  $("stop-label").textContent=run?.status==="stopping"?"正在保存…":"停止并保存";
  $("resume-button").hidden=!run||!["stopped","failed","interrupted"].includes(run.status)||!run.checkpoint_ready||run.step>=run.total;
  $("resume-button").disabled=busy||pending||!online;
  $("shutdown-button").disabled=busy||pending||!online;
  $("shutdown-button").title=busy?"请先停止并保存当前训练":"关闭本机界面服务";
  $("active-notice").hidden=!trainingBusy||state.run?.id===state.active_id;
  $("data-prepare").disabled=busy||pending||!online;
  for(const id of ["web-docs","chat-rows","tool-rows"])$(id).disabled=busy||pending;
  $("data-cancel").hidden=!dataBusy;$("data-cancel").disabled=state.data_job?.status==="cancelling"||pending||!online;
}
function renderDataJob(){
  const job=state.data_job||{status:"idle"};
  $("data-preparation").hidden=job.status==="idle";$("data-preparation").dataset.status=job.status;
  $("data-prepare").textContent=boot?.datasets.some(d=>d.source_type==="public")?"再准备一份":"获取数据";
  $("data-message").textContent=job.message||"等待准备";
  if(Number.isFinite(job.current)&&Number.isFinite(job.total)&&job.total>0)$("data-progress").value=Math.min(100,job.current/job.total*100);else $("data-progress").removeAttribute("value");
  if(job.status==="completed") {const r=job.result;$("data-progress").value=100;$("data-detail").textContent=r?`网页 ${number(r.accepted.web)} 篇；已编码对话 ${number(r.prepared.sft.train.documents+r.prepared.sft.val.documents)} 条。`:"可在训练数据中选择公开数据。";}
  else if(job.phase?.endsWith("_download")&&job.unit==="bytes"&&Number.isFinite(job.current))$("data-detail").textContent=`本阶段已下载 ${(job.current/1048576).toFixed(1)} MiB${job.total?` / ${(job.total/1048576).toFixed(1)} MiB`:""}`;
  else if(job.phase==="rate_limit")$("data-detail").textContent="已下载的缓存会保留，等待后自动继续。";
  else $("data-detail").textContent=Number.isFinite(job.current)&&Number.isFinite(job.total)?`本阶段 ${number(job.current)} / ${number(job.total)}`:"分词与编码阶段的耗时取决于语料大小。";
}
function render() {
  const run=state.run, latest=run?.latest||{};
  const runDataset=boot?.datasets.find(d=>d.path===run?.params?.data);
  $("run-context").hidden=!run;$("run-context").textContent=run?`当前查看：${runDataset?.display_name||(runDataset?.source_type==="demo"?"示例数据":run.params.data)} · ${run.params.stage==="sft"?"对话与工具训练":"从零预训练"}`:"";
  const pct=run?.progress||0;
  $("progress-percent").replaceChildren(document.createTextNode(String(Math.round(pct*10)/10)));
  const percentSign=document.createElement("span");percentSign.textContent="%";$("progress-percent").append(percentSign);
  $("training-progress").value=pct; $("training-progress").textContent=`${pct}%`;
  $("step-count").textContent=run?`${number(run.step)} / ${number(run.total)} 步`:`0 / ${number(Number($("steps").value)||100)} 步`;
  $("status-badge").textContent=run?(statusNames[run.status]||run.status):"准备就绪";
  $("status-badge").dataset.status=run?.status||"idle";
  $("status-badge").closest(".progress-panel").dataset.complete=String(run?.status==="completed");
  $("eta-label").textContent=run?.status==="completed"?"本次训练":"预计剩余";
  $("eta").textContent=!run?"等待开始":run.status==="completed"?"计划已完成":run.status==="stopping"?"正在保存权重":run.status==="stopped"?"可继续训练":ACTIVE.has(run.status)?(run.eta_seconds==null?"测量中…":`约 ${duration(run.eta_seconds)}`):"—";
  let caption="设置好参数后，点击左侧「开始训练」。";
  if(run) {
    caption={starting:"正在加载数据、创建模型，第一步结束后进度会更新。",running:"进度来自训练器的实际记录；权重每 25 步及结束时保存。",stopping:"等待当前一步结束，并保存模型、优化器与训练进度。",stopped:"权重和训练状态已保存，点击「继续训练」完成剩余步数。",completed:"本次计划已完成，最终权重已保存。",failed:run.error||"训练进程退出，请查看下方日志。",interrupted:"界面服务曾中断，可从已保存的权重继续训练。"}[run.status]||"";
  }
  $("progress-caption").textContent=caption;
  $("loss").textContent=loss(latest.loss);
  const lastValidation=(run?.series||[]).filter(p=>p.validation_loss!==undefined).at(-1)?.validation_loss ?? run?.summary?.validation_loss;
  $("validation-loss").textContent=loss(lastValidation);
  $("speed").textContent=run?.tokens_per_second==null?"—":number(Math.round(run.tokens_per_second));
  $("elapsed").textContent=run&&run.elapsed_seconds>0?duration(run.elapsed_seconds):"—";
  $("output-row").hidden=!run;$("output-path").textContent=run?.output||"";
  $("chart-empty").hidden=Boolean(run?.series?.length);
  const logs=run?.logs?.join("\n") || "训练启动后，输出会显示在这里。";
  if(logs!==lastLog) {
    const pre=$("logs"), nearBottom=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-35;
    pre.textContent=logs;if(nearBottom||!lastLog) pre.scrollTop=pre.scrollHeight;
    lastLog=logs;
  }
  $("log-count").textContent=run?`${run.logs?.length||0} 行输出`:"等待输出";
  renderHistory();renderDataJob();renderControls();drawChart();
}
function renderHistory() {
  $("history-count").textContent=`${state.runs.length} 次实验`;$("history-empty").hidden=state.runs.length>0;
  $("history-body").replaceChildren();
  for(const run of state.runs) {
    const row=document.createElement("tr");row.classList.toggle("is-selected",run.id===state.run?.id);
    const time=new Intl.DateTimeFormat("zh-CN",{timeZone:"Asia/Shanghai",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}).format(new Date(run.created_at));
    for(const value of [time,run.params.stage==="sft"?"对话与工具":"从零预训练",`${number(run.step)} / ${number(run.total)}`]) {const td=document.createElement("td");td.textContent=value;row.append(td);}
    const statusCell=document.createElement("td"), badge=document.createElement("span");badge.className="status-badge";badge.dataset.status=run.status;badge.textContent=statusNames[run.status]||run.status;statusCell.append(badge);row.append(statusCell);
    const actionCell=document.createElement("td"), button=document.createElement("button");button.className="text-button";button.textContent=run.id===state.run?.id?"查看中":"查看";button.addEventListener("click",()=>{selectedId=run.id;poll(true);});actionCell.append(button);row.append(actionCell);$("history-body").append(row);
  }
}
function drawChart() {
  const canvas=$("loss-chart"), box=$("chart-container");
  const width=box.clientWidth,height=box.clientHeight,dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  const c=canvas.getContext("2d");c.scale(dpr,dpr);
  const left=38,right=14,top=16,bottom=30,w=width-left-right,h=height-top-bottom;
  const series=(state.run?.series||[]).filter(p=>Number.isFinite(p.loss));
  const values=series.flatMap(p=>[p.loss,p.validation_loss]).filter(Number.isFinite);
  const high=values.length?Math.max(...values):8,low=values.length?Math.min(...values):0;
  const spread=Math.max(high-low,.2),yMin=Math.max(0,low-spread*.13),yMax=high+spread*.13;
  const minStep=series[0]?.step||0,maxStep=Math.max((series.at(-1)?.step||0),minStep+1);
  const x=n=>left+(n-minStep)/(maxStep-minStep)*w,y=n=>top+(yMax-n)/(yMax-yMin)*h;
  c.font='10px "Segoe UI", sans-serif';c.fillStyle="#aab6c8";c.strokeStyle="#edf1f7";c.lineWidth=1;
  for(let i=0;i<=4;i++) {const yy=top+i*h/4;c.beginPath();c.moveTo(left,yy+.5);c.lineTo(width-right,yy+.5);c.stroke();c.textAlign="right";c.fillText((yMax-i*(yMax-yMin)/4).toFixed(1),left-10,yy+3);}
  c.textAlign="center";
  for(let i=0;i<=4;i++) {const xx=left+i*w/4;const value=series.length?Math.round(minStep+i*(maxStep-minStep)/4):Math.round(i*(Number($("steps").value)||100)/4);c.fillText(String(value),xx,height-10);}
  if(!series.length) return;
  function line(key,color,fill=false) {
    const points=series.filter(p=>Number.isFinite(p[key]));if(!points.length) return;
    if(fill&&points.length>1) {c.beginPath();c.moveTo(x(points[0].step),top+h);for(const p of points)c.lineTo(x(p.step),y(p[key]));c.lineTo(x(points.at(-1).step),top+h);c.closePath();c.fillStyle="#235ed608";c.fill();}
    c.beginPath();points.forEach((p,i)=>i?c.lineTo(x(p.step),y(p[key])):c.moveTo(x(p.step),y(p[key])));c.strokeStyle=color;c.lineWidth=1.8;c.lineJoin="round";c.stroke();
    if(key==="validation_loss"||points.length===1)for(const p of points){c.beginPath();c.arc(x(p.step),y(p[key]),2.5,0,Math.PI*2);c.fillStyle=color;c.fill();}
  }
  line("loss","#235ed6",true);line("validation_loss","#cc8a26");
}
async function poll(manual=false) {
  if(closed) return;
  try {
    const previousStatus=state.run?.status;
    const previousId=state.run?.id;
    const previousDataStatus=state.data_job?.status;
    state=await request(`/api/state${selectedId?`?run_id=${encodeURIComponent(selectedId)}`:""}`);
    if(state.active_id&&state.run?.id===state.active_id&&previousId!==state.run.id) applyParams(state.run.params);
    setOnline(true);render();
    if(["completed","stopped","failed"].includes(state.run?.status)&&previousStatus!==state.run.status) bootstrap().catch(()=>{});
    if(state.data_job?.status==="completed"&&previousDataStatus!=="completed") {
      await bootstrap();const result=state.data_job.result;const preferred=result?.outputs?.[$("stage").value];
      if(preferred)$("dataset").value=preferred;updateHints();notify("公开数据已准备好，已选中新的数据集。");
    }
  } catch(error) {setOnline(false);renderControls();if(manual)notify(error.message,"error");}
}
async function tick(){await poll();if(!closed)setTimeout(tick,document.hidden?1800:650);}
function applyParams(params) {
  $("stage").value=params.stage;stageChanged(false);$("model").value=params.model;$("dataset").value=params.data;
  for(const [id,key] of [["steps","steps"],["learning-rate","learning_rate"],["threads","threads"],["batch-size","batch_size"],["accumulation","accumulation"]])$(id).value=params[key];
  if(params.stage==="pretrain") $("block-size").value=params.block_size;
  if(params.init_from)$("checkpoint").value=params.init_from;updateHints();
}
$("train-form").addEventListener("submit",async event=>{
  event.preventDefault();if(pending||state.active_id)return;
  pending=true;renderControls();
  try {
    const body={stage:$("stage").value,model:$("model").value,data:$("dataset").value,steps:Number($("steps").value),learning_rate:Number($("learning-rate").value),threads:Number($("threads").value),block_size:Number($("block-size").value),batch_size:Number($("batch-size").value),accumulation:Number($("accumulation").value),init_from:$("checkpoint").value};
    const run=await request("/api/train",body);selectedId=run.id;state.active_id=run.id;state.run=run;render();notify("训练已启动，第一步完成后进度会更新。");await poll(true);
  } catch(error){notify(error.message,"error");} finally{pending=false;renderControls();}
});
$("stop-button").addEventListener("click",async()=>{
  pending=true;renderControls();try{await request("/api/stop",{});await poll(true);notify("已请求停止，正在等待当前一步结束并保存。");}catch(error){notify(error.message,"error");}finally{pending=false;renderControls();}
});
$("resume-button").addEventListener("click",async()=>{
  pending=true;renderControls();try{const run=await request("/api/resume",{run_id:state.run.id});applyParams(run.params);selectedId=run.id;state.active_id=run.id;state.run=run;render();notify("继续上次训练，沿用原参数。");await poll(true);}catch(error){notify(error.message,"error");}finally{pending=false;renderControls();}
});
$("go-active").addEventListener("click",()=>{selectedId=state.active_id;poll(true);});
$("stage").addEventListener("change",()=>stageChanged());
for(const id of ["dataset","model","checkpoint"])$(id).addEventListener("change",updateHints);
for(const id of ["steps","batch-size","accumulation","block-size"])$(id).addEventListener("input",()=>{updateHints();if(!state.run){render();}});
$("data-prepare").addEventListener("click",async()=>{
  for(const id of ["web-docs","chat-rows","tool-rows"])if(!$(id).reportValidity())return;
  pending=true;renderControls();try{state.data_job=await request("/api/data/prepare",{web_docs:Number($("web-docs").value),chat_rows:Number($("chat-rows").value),tool_rows:Number($("tool-rows").value)});render();notify("已开始准备公开数据，完成后可直接用于训练。");}catch(error){notify(error.message,"error");}finally{pending=false;renderControls();}
});
$("data-cancel").addEventListener("click",async()=>{try{state.data_job=await request("/api/data/cancel",{});render();}catch(error){notify(error.message,"error");}});
$("copy-path").addEventListener("click",async()=>{try{await navigator.clipboard.writeText(`${boot.workspace}\\${state.run.output.replaceAll("/","\\")}`);notify("已复制输出目录。");}catch{notify("请选中上方输出路径复制。","error");}});
$("help-button").addEventListener("click",()=>$("help-dialog").showModal());$("help-close").addEventListener("click",()=>$("help-dialog").close());
$("toast-close").addEventListener("click",()=>$("toast").hidden=true);
$("shutdown-button").addEventListener("click",async()=>{try{await request("/api/shutdown",{});closed=true;setOnline(false);renderControls();notify("界面服务已关闭。双击「启动训练界面.cmd」可重新打开。");}catch(error){notify(error.message,"error");}});
new ResizeObserver(drawChart).observe($("chart-container"));
(async()=>{try{await bootstrap(true);setOnline(true);render();tick();}catch(error){setOnline(false);notify(`无法加载训练界面：${error.message}`,"error");}})();
