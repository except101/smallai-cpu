"use strict";
const $ = id => document.getElementById(id);
const ACTIVE = new Set(["starting", "running", "stopping"]);
const statusNames = {starting:"正在准备",running:"训练中",stopping:"正在保存",stopped:"已停止",completed:"已完成",failed:"训练失败",interrupted:"服务曾中断"};
let boot = null, state = {active_id:null,run:null,runs:[],data_job:{status:"idle"}}, selectedId = null, online = false, pending = false, closed = false, toastTimer = null, lastLog = "";
let importPreview = null, importConsumed = false, uploading = false;
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
async function request(path, body, retried=false) {
  const options = body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json","X-SmallAI-Token":boot.csrf_token},body:JSON.stringify(body)};
  const response=await fetch(path,options);
  const value=await response.json();
  if(response.status===403&&body!==undefined&&!retried&&boot) {
    const fresh=await request("/api/bootstrap");boot.csrf_token=fresh.csrf_token;
    return request(path,body,true);
  }
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
  renderLibrary(); populateImportTokenizers();
  if(first && !boot.datasets.length) showView("data");
}
function stageChanged(resetRate=true) {
  if(!boot) return;
  const sft=$("stage").value==="sft", oldDataset=$("dataset").value;
  $("model-field").hidden=sft; $("checkpoint-field").hidden=!sft; $("block-field").hidden=sft; $("sft-block-hint").hidden=!sft;
  $("stage-hint").textContent=sft?"在自己的权重上学习题库问答与工具调用。":"从随机权重开始，学习文本规律。";
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
  $("checkpoint-hint").textContent=cp?`${(cp.parameters/10000).toLocaleString("zh-CN",{maximumFractionDigits:1})} 万参数 · 分词器匹配`:"没有兼容权重，请先选择同一份数据的文本做从零预训练。";
  $("dataset-hint").textContent=data?`${data.source_type==="question_bank"?"导入题库 · ":data.source_type==="public"?"公开数据 · ":"流程示例 · "}训练 ${number(data.splits.train.documents)} 条 · 验证 ${number(data.splits.val.documents)} 条` : "还没有可用数据，打开「数据与题库」导入或获取。";
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
  const enoughTokens=!selectedData||selectedData.stage!=="pretrain"||Math.min(selectedData.splits.train.tokens,selectedData.splits.val.tokens)>Number($("block-size").value);
  const ready=selectedData?.tokenizer_ready && enoughTokens && $("dataset").value && ($("stage").value!=="sft" || $("checkpoint").value);
  $("start-button").disabled=!online||busy||pending||!ready;
  $("start-label").textContent=dataBusy?"正在准备数据":trainingBusy?"训练正在进行":pending?"请稍候…":"开始训练";
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
  $("question-file").disabled=busy||pending||!online;
  $("drop-zone").dataset.disabled=String(busy||pending||!online);
  $("import-prepare").disabled=busy||pending||!online||!importPreview||importConsumed;
  $("import-prepare").textContent=importConsumed?"已开始准备":"导入并准备训练";
  $("import-tokenizer").disabled=busy||pending||importConsumed;
  for(const button of document.querySelectorAll("[data-steps]")) {
    button.disabled=busy||pending||closed;
    button.setAttribute("aria-pressed",String(Number(button.dataset.steps)===Number($("steps").value)));
  }
  let reason="数据和训练配置已就绪，可以开始训练。";
  if(closed) reason="服务已关闭，请重新启动训练界面。";
  else if(!online) reason="正在连接本机服务，请稍候。";
  else if(dataBusy) reason="正在准备数据，完成后可以开始训练。";
  else if(trainingBusy) reason="训练正在进行，可以停止并保存后调整设置。";
  else if(!selectedData) reason="先打开「数据与题库」，导入题库或获取公开数据。";
  else if(!selectedData.tokenizer_ready) reason="这份数据的分词器缺失，请重新准备数据。";
  else if(!enoughTokens) reason="数据量较少，请在「更多设置」中减小文本块长度，或导入更多题目。";
  else if($("stage").value==="sft"&&!$("checkpoint").value) reason="还没有兼容权重。请先用同一份数据的文本做从零预训练。";
  $("readiness").textContent=reason;$("readiness").dataset.ready=String(Boolean(ready)&&!busy&&online&&!closed);
  $("flow-data").dataset.ready=String(Boolean(selectedData));
  $("flow-data-detail").textContent=boot?.datasets.length?`${boot.datasets.length} 份可选数据 · 可继续导入题库`:"导入题库或获取公开数据";
  $("flow-model").dataset.ready=String(Boolean(ready));
  $("flow-model-detail").textContent=$("stage").value==="sft"?"使用自己的权重学习问答":"从随机权重开始预训练";
  $("flow-train").dataset.ready=String(run?.status==="completed");
  $("flow-train-detail").textContent=trainingBusy?"训练进行中 · 可停止并保存":run?.status==="completed"?"最近一次训练已完成并保存":"实时进度 · 可停止续训";
  for(const row of document.querySelectorAll(".dataset-row")) {
    const chosen=row.dataset.path===$("dataset").value;
    row.classList.toggle("is-selected",chosen);
    const button=row.querySelector("button");button.disabled=busy||pending||!online;button.textContent=chosen?"已选中":"用于训练";
  }
}
function renderDataJob(){
  const job=state.data_job||{status:"idle"};
  $("data-preparation").hidden=job.status==="idle";$("data-preparation").dataset.status=job.status;
  $("data-prepare").textContent=boot?.datasets.some(d=>d.source_type==="public")?"再准备一份":"获取数据";
  $("data-message").textContent=job.message||"等待准备";
  if(Number.isFinite(job.current)&&Number.isFinite(job.total)&&job.total>0)$("data-progress").value=Math.min(100,job.current/job.total*100);else $("data-progress").removeAttribute("value");
  if(job.status==="completed") {
    const r=job.result;$("data-progress").value=100;
    $("data-detail").textContent=r?.kind==="question_bank"?`${r.filename} · 可用 ${number(r.accepted)} 道 · 超长跳过 ${number(r.too_long_skipped)} 道。已生成题库文本与问答数据。`:r?`网页 ${number(r.accepted.web)} 篇；已编码对话 ${number(r.prepared.sft.train.documents+r.prepared.sft.val.documents)} 条。`:"数据已准备好，可以选择用于训练。";
  }
  else if(["failed","cancelled","interrupted"].includes(job.status)) {$("data-progress").value=0;$("data-detail").textContent=job.status==="failed"?"调整数据后重新导入，或更换分词器重试。":"可重新选择文件继续准备。";}
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
  if(width<1||height<1)return;
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
    if(state.data_job?.kind==="question_bank"&&["failed","cancelled"].includes(state.data_job.status))importConsumed=false;
    if(state.active_id&&state.run?.id===state.active_id&&previousId!==state.run.id) applyParams(state.run.params);
    setOnline(true);render();
    if(["completed","stopped","failed"].includes(state.run?.status)&&previousStatus!==state.run.status) bootstrap().catch(()=>{});
    if(state.data_job?.status==="completed"&&previousDataStatus!=="completed") {
      await bootstrap();const result=state.data_job.result;
      if(result?.kind==="question_bank") {
        const data=boot.datasets.find(d=>d.path===result.outputs.sft);
        $("stage").value=boot.checkpoints.some(c=>c.tokenizer_sha256===data?.tokenizer_sha256)?"sft":"pretrain";
        stageChanged();
      }
      const preferred=result?.outputs?.[$("stage").value];
      if(preferred)$("dataset").value=preferred;updateHints();
      notify(result?.kind==="question_bank"?`题库已准备好，已选中${$("stage").value==="sft"?"问答":"预训练"}数据。`:"公开数据已准备好，已选中新的数据集。");
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
    const run=await request("/api/train",body);selectedId=run.id;state.active_id=run.id;state.run=run;showView("monitor");render();notify("训练已启动，第一步完成后进度会更新。");await poll(true);
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

function showView(name) {
  for(const view of ["monitor","data"]) {
    const selected=view===name;
    $(`${view}-view`).hidden=!selected;
    $(`tab-${view}`).setAttribute("aria-selected",String(selected));
    $(`tab-${view}`).tabIndex=selected?0:-1;
  }
  if(name==="monitor") requestAnimationFrame(drawChart);
}
for(const view of ["monitor","data"]) {
  $(`tab-${view}`).addEventListener("click",()=>showView(view));
  $(`tab-${view}`).addEventListener("keydown",event=>{
    if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
    event.preventDefault();const target=event.key==="Home"?"monitor":event.key==="End"?"data":view==="data"?"monitor":"data";
    showView(target);$(`tab-${target}`).focus();
  });
}
$("open-data-link").addEventListener("click",event=>{event.preventDefault();showView("data");$("tab-data").focus();});
for(const button of document.querySelectorAll("[data-steps]"))button.addEventListener("click",()=>{
  $("steps").value=button.dataset.steps;updateHints();if(!state.run)render();
});

function renderLibrary() {
  $("dataset-count").textContent=String(boot.datasets.length);
  $("dataset-library").replaceChildren();
  if(!boot.datasets.length) {
    const empty=document.createElement("p");empty.className="library-empty";empty.textContent="还没有可训练的数据。先导入上方题库，或在下方获取公开数据。";$("dataset-library").append(empty);
  }
  for(const data of boot.datasets) {
    const row=document.createElement("div");row.className="dataset-row";row.dataset.path=data.path;
    const body=document.createElement("div"),title=document.createElement("strong"),detail=document.createElement("span");
    title.textContent=data.display_name|| (data.source_type==="demo"?`流程示例 · ${data.stage==="sft"?"问答与工具":"中文语料"}`:data.path);
    detail.textContent=`${data.stage==="sft"?"问答训练":"从零预训练"} · 训练 ${number(data.splits.train.documents)} 条 / 验证 ${number(data.splits.val.documents)} 条`;
    body.append(title,detail);
    const button=document.createElement("button");button.type="button";button.className="secondary-button";button.textContent="用于训练";
    button.addEventListener("click",()=>{$("stage").value=data.stage;stageChanged();$("dataset").value=data.path;updateHints();showView("monitor");});
    row.append(body,button);$("dataset-library").append(row);
  }
  renderControls();
}
function populateImportTokenizers() {
  const previous=$("import-tokenizer").value, selected=boot.datasets.find(d=>d.path===$("dataset").value), seen=new Set();
  $("import-tokenizer").replaceChildren();
  for(const data of boot.datasets) {
    if(!data.tokenizer_ready||seen.has(data.tokenizer_sha256))continue;
    seen.add(data.tokenizer_sha256);
    addOption($("import-tokenizer"),data.tokenizer_path,`沿用：${data.display_name|| (data.source_type==="demo"?"示例模型":data.path)}`);
  }
  addOption($("import-tokenizer"),"new","建立新分词器，从零训练");
  if([...$("import-tokenizer").options].some(option=>option.value===previous))$("import-tokenizer").value=previous;
  else if(selected?.tokenizer_ready)$("import-tokenizer").value=selected.tokenizer_path;
  importTokenizerHint();
}
function importTokenizerHint() {
  $("import-tokenizer-hint").textContent=$("import-tokenizer").value==="new"?"会用题库训练自己的新分词器，之后需要先做从零预训练。":"保留已有模型使用的词表，准备后可选择兼容的权重继续学习题库。";
}
$("import-tokenizer").addEventListener("change",importTokenizerHint);

async function uploadQuestionFile(file) {
  if(!file)return;
  if(pending||state.active_id||["running","cancelling"].includes(state.data_job?.status)||!online)return;
  $("import-error").hidden=true;$("import-preview").hidden=true;importPreview=null;importConsumed=false;
  if(file.size>10*1024*1024||!file.size) {$("import-error").textContent="文件为空或超过 10 MiB，请拆分后重新选择。";$("import-error").hidden=false;return;}
  pending=true;uploading=true;renderControls();$("drop-zone").querySelector("strong").textContent="正在读取和检查题库…";
  try {
    const bytes=new Uint8Array(await file.arrayBuffer());let binary="";
    for(let offset=0;offset<bytes.length;offset+=32768)binary+=String.fromCharCode(...bytes.subarray(offset,offset+32768));
    importPreview=await request("/api/questions/preview",{filename:file.name,content:btoa(binary)});
    $("import-filename").textContent=importPreview.filename;
    for(const [id,key] of [["import-valid","accepted"],["import-duplicates","duplicates"],["import-invalid","invalid"]])$(id).textContent=number(importPreview[key]);
    $("question-samples").replaceChildren();
    for(const [index,record] of importPreview.samples.entries()) {
      const article=document.createElement("article"),label=document.createElement("span"),question=document.createElement("p"),answer=document.createElement("p");
      label.className="sample-label";label.textContent=`预览 ${index+1}`;
      question.className="sample-question";question.textContent=record.messages.filter(m=>m.role==="user").map(m=>m.content).join("\n").slice(0,2000);
      const last=record.messages.at(-1);answer.className="sample-answer";answer.textContent=(last.content||JSON.stringify(last.tool_calls)).slice(0,2000);
      article.append(label,question,answer);$("question-samples").append(article);
    }
    $("import-errors").replaceChildren();$("import-errors").hidden=!importPreview.errors.length;
    for(const error of importPreview.errors) {const item=document.createElement("li");item.textContent=`第 ${error.row} 行：${error.message}`;$("import-errors").append(item);}
    populateImportTokenizers();$("import-preview").hidden=false;
  } catch(error) {$("import-error").textContent=error.message;$("import-error").hidden=false;}
  finally {pending=false;uploading=false;$("drop-zone").querySelector("strong").textContent="拖入题库文件，或点击选择";renderControls();}
}
$("question-file").addEventListener("change",event=>{uploadQuestionFile(event.target.files[0]);event.target.value="";});
for(const name of ["dragover","dragenter"])$("drop-zone").addEventListener(name,event=>{event.preventDefault();if(!$("question-file").disabled)$("drop-zone").classList.add("is-dragging");});
for(const name of ["dragleave","drop"])$("drop-zone").addEventListener(name,event=>{event.preventDefault();$("drop-zone").classList.remove("is-dragging");if(name==="drop")uploadQuestionFile(event.dataTransfer.files[0]);});
$("drop-zone").addEventListener("click",event=>{if(event.target.closest("label,input"))return;if(!$("question-file").disabled)$("question-file").click();});
$("import-prepare").addEventListener("click",async()=>{
  if(!importPreview||pending||importConsumed)return;
  pending=true;renderControls();$("import-error").hidden=true;
  try {
    state.data_job=await request("/api/questions/prepare",{import_id:importPreview.id,tokenizer_path:$("import-tokenizer").value==="new"?null:$("import-tokenizer").value});
    importConsumed=true;render();notify("已开始准备题库，进度显示在下方。");
  } catch(error) {$("import-error").textContent=error.message;$("import-error").hidden=false;}
  finally {pending=false;renderControls();}
});
new ResizeObserver(drawChart).observe($("chart-container"));
(async()=>{try{await bootstrap(true);setOnline(true);render();tick();}catch(error){setOnline(false);notify(`无法加载训练界面：${error.message}`,"error");}})();
